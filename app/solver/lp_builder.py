"""把“原料库 + 配方规格”翻译成单纯形 LP，并解释回业务结构。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simplex import Row

# 干物质作为内置营养项，始终存在
DM_CODE = "DM"


@dataclass(frozen=True)
class Ingredient:
    code: str
    name: str
    price: float
    nutrients: dict[str, float]  # 营养代码 -> 含量（小数，0.08 表示 8%）


@dataclass(frozen=True)
class NutrientBound:
    code: str
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class RatioConstraint:
    """分子营养 / 分母营养 ∈ [lower, upper]。"""

    numerator: str
    denominator: str
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class IngredientLimit:
    code: str
    min_ratio: float = 0.0
    max_ratio: float = 1.0


@dataclass(frozen=True)
class RecipeSpec:
    code: str
    name: str
    ingredients: tuple[IngredientLimit, ...]
    nutrients: tuple[NutrientBound, ...] = ()
    ratios: tuple[RatioConstraint, ...] = ()
    total_mass: float = 1.0


@dataclass
class BuiltLP:
    ingredients: list[str]  # 列下标 -> 原料代码（只含配方引用的原料）
    rows: list[Row]
    costs: np.ndarray
    # 业务视图：每种约束的描述，供结果解释/诊断使用
    constraint_info: dict[str, dict] = field(default_factory=dict)
    nutrient_matrix: np.ndarray | None = None
    nutrient_codes: list[str] = field(default_factory=list)
    ratio_rows: list[str] = field(default_factory=list)


class BuildError(ValueError):
    def __init__(self, field_: str, message: str):
        super().__init__(f"{field_}: {message}")
        self.field = field_
        self.message = message


def build_lp(
    spec: RecipeSpec,
    catalog: dict[str, Ingredient],
    all_nutrients: set[str],
) -> BuiltLP:
    """构建 LP。catalog 为当前原料库代码->原料；all_nutrients 为库中全部营养项。

    校验（返回带字段名的错误）：引用不存在的原料/营养项、比值分母全为零等。
    数值本身（负价格、非有限数、上下限颠倒等）在 validation 层校验。
    """

    info: dict[str, dict] = {}
    codes: list[str] = []
    for il in spec.ingredients:
        if il.code not in catalog:
            raise BuildError(f"ingredients[{il.code}]", f"原料不存在: {il.code}")
        codes.append(il.code)
    n = len(codes)
    idx = {c: i for i, c in enumerate(codes)}
    costs = np.array([catalog[c].price for c in codes], dtype=float)

    rows: list[Row] = []

    def add(key: str, a, sense, b, meta):
        rows.append(Row(key, np.asarray(a, dtype=float), sense, float(b), meta))
        info[key] = meta

    # 1) 总量 = 1 kg（非默认总量时 = total_mass）
    add(
        "total",
        np.ones(n),
        "=",
        spec.total_mass,
        {"kind": "total", "label": f"总量 = {spec.total_mass} kg"},
    )

    # 2) 原料添加比例上下限
    for il in spec.ingredients:
        j = idx[il.code]
        e = np.zeros(n)
        e[j] = 1.0
        if il.min_ratio > 0.0:
            add(
                f"ing_min:{il.code}",
                e,
                ">=",
                il.min_ratio,
                {
                    "kind": "ingredient_min",
                    "ingredient": il.code,
                    "label": f"{il.code} 最小添加 {il.min_ratio:g}",
                },
            )
        if il.max_ratio < 1.0:
            add(
                f"ing_max:{il.code}",
                e,
                "<=",
                il.max_ratio,
                {
                    "kind": "ingredient_max",
                    "ingredient": il.code,
                    "label": f"{il.code} 最大添加 {il.max_ratio:g}",
                },
            )

    # 3) 营养上下限（系数为每 kg 配方中该营养的含量）
    # 营养矩阵覆盖库中全部营养项，即使只出现在比值约束里也要能在结果中汇报
    nutrient_codes = sorted(all_nutrients)
    N = np.zeros((len(nutrient_codes), n))
    for j, code in enumerate(codes):
        vals = catalog[code].nutrients
        for r_, ncode in enumerate(nutrient_codes):
            N[r_, j] = vals.get(ncode, 0.0)
    nrow = {nc: r_ for r_, nc in enumerate(nutrient_codes)}
    for nb in spec.nutrients:
        if nb.code not in all_nutrients:
            raise BuildError(f"nutrients[{nb.code}]", f"营养项不存在: {nb.code}")
        a = N[nrow[nb.code]]
        if nb.lower is not None and nb.lower > 0.0:
            add(
                f"nut_min:{nb.code}",
                a,
                ">=",
                nb.lower,
                {
                    "kind": "nutrient_min",
                    "nutrient": nb.code,
                    "label": f"{nb.code} ≥ {nb.lower:g}",
                },
            )
        if nb.upper is not None:
            add(
                f"nut_max:{nb.code}",
                a,
                "<=",
                nb.upper,
                {
                    "kind": "nutrient_max",
                    "nutrient": nb.code,
                    "label": f"{nb.code} ≤ {nb.upper:g}",
                },
            )

    # 4) 比值约束 sum(a_num*x)/sum(a_den*x) ∈ [lo, hi]
    #    线性化为  sum((a_num - lo*a_den)*x) >= 0
    #             sum((a_num - hi*a_den)*x) <= 0
    ratio_rows: list[str] = []
    for k, rc in enumerate(spec.ratios):
        if rc.numerator not in all_nutrients:
            raise BuildError(
                f"ratios[{k}].numerator", f"营养项不存在: {rc.numerator}"
            )
        if rc.denominator not in all_nutrients:
            raise BuildError(
                f"ratios[{k}].denominator", f"营养项不存在: {rc.denominator}"
            )
        a_num = np.array(
            [catalog[c].nutrients.get(rc.numerator, 0.0) for c in codes]
        )
        a_den = np.array(
            [catalog[c].nutrients.get(rc.denominator, 0.0) for c in codes]
        )
        if not np.any(np.abs(a_den) > 0.0):
            raise BuildError(
                f"ratios[{k}].denominator",
                f"比值约束分母营养 {rc.denominator} 在所有引用原料中均为零",
            )
        if rc.lower is not None:
            key = f"ratio_lo:{rc.numerator}/{rc.denominator}"
            add(
                key,
                a_num - rc.lower * a_den,
                ">=",
                0.0,
                {
                    "kind": "ratio_lower",
                    "numerator": rc.numerator,
                    "denominator": rc.denominator,
                    "bound": rc.lower,
                    "label": f"{rc.numerator}/{rc.denominator} ≥ {rc.lower:g}",
                },
            )
            ratio_rows.append(key)
        if rc.upper is not None:
            key = f"ratio_hi:{rc.numerator}/{rc.denominator}"
            add(
                key,
                a_num - rc.upper * a_den,
                "<=",
                0.0,
                {
                    "kind": "ratio_upper",
                    "numerator": rc.numerator,
                    "denominator": rc.denominator,
                    "bound": rc.upper,
                    "label": f"{rc.numerator}/{rc.denominator} ≤ {rc.upper:g}",
                },
            )
            ratio_rows.append(key)

    return BuiltLP(
        ingredients=codes,
        rows=rows,
        costs=costs,
        constraint_info=info,
        nutrient_matrix=N,
        nutrient_codes=nutrient_codes,
        ratio_rows=ratio_rows,
    )
