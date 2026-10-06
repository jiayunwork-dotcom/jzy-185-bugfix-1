"""业务校验：营养值/价格/上下限/比例/引用合法性，错误全部带字段名。"""

from __future__ import annotations

import math

from ..schemas import (
    IngredientIn,
    NutrientBoundIn,
    RatioIn,
    RecipeIn,
    IngredientLimitIn,
)


class ValidationErrors(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))
        self.errors = errors


def _finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def validate_ingredients(
    ingredients: list[IngredientIn], known_nutrients: set[str]
):
    errors: list[dict] = []
    codes: set[str] = set()
    for i, ing in enumerate(ingredients):
        base = f"ingredients[{i}]"
        if ing.code in codes:
            errors.append({"field": f"{base}.code", "message": "原料代码重复"})
        codes.add(ing.code)
        if not _finite(ing.price):
            errors.append(
                {"field": f"{base}.price", "message": "单价必须为有限数"}
            )
        elif ing.price < 0:
            errors.append(
                {"field": f"{base}.price", "message": "单价不能为负"}
            )
        for ncode, val in ing.nutrients.items():
            fld = f"{base}.nutrients.{ncode}"
            if ncode not in known_nutrients:
                errors.append({"field": fld, "message": f"营养项不存在: {ncode}"})
            elif not _finite(val):
                errors.append({"field": fld, "message": "营养含量必须为有限数"})
            elif val < 0:
                errors.append({"field": fld, "message": "营养含量不能为负"})
        # 干物质必须给出（含量库以小数表示；DM 缺省视为 1 不严谨，直接报错提示）
        if "DM" in known_nutrients and "DM" not in ing.nutrients:
            errors.append(
                {"field": f"{base}.nutrients.DM", "message": "干物质含量必填"}
            )
    if not ingredients:
        errors.append({"field": "ingredients", "message": "原料库不能为空"})
    if errors:
        raise ValidationErrors(errors)


def validate_recipe(spec: RecipeIn, known_nutrients: set[str], ingredient_codes: set[str]):
    errors: list[dict] = []
    seen: set[str] = set()

    def check_ratio_range(prefix: str, lo, hi):
        if lo is not None and not _finite(lo):
            errors.append({"field": f"{prefix}.lower", "message": "必须为有限数"})
        if hi is not None and not _finite(hi):
            errors.append({"field": f"{prefix}.upper", "message": "必须为有限数"})
        if (
            lo is not None
            and hi is not None
            and _finite(lo)
            and _finite(hi)
            and lo > hi
        ):
            errors.append(
                {"field": f"{prefix}", "message": "下限不能高于上限"}
            )
        if lo is not None and _finite(lo) and lo < 0:
            errors.append({"field": f"{prefix}.lower", "message": "比值下限不能为负"})

    if not spec.ingredients:
        errors.append({"field": "ingredients", "message": "配方至少包含一种原料"})
    for i, il in enumerate(spec.ingredients):
        prefix = f"ingredients[{i}]"
        if il.code in seen:
            errors.append(
                {"field": f"{prefix}.code", "message": "配方内原料重复"}
            )
        seen.add(il.code)
        if il.code not in ingredient_codes:
            errors.append(
                {"field": f"{prefix}.code", "message": f"引用了不存在的原料: {il.code}"}
            )
        for name, v in (("min_ratio", il.min_ratio), ("max_ratio", il.max_ratio)):
            if not _finite(v):
                errors.append({"field": f"{prefix}.{name}", "message": "必须为有限数"})
        if _finite(il.min_ratio) and not (0.0 <= il.min_ratio <= 1.0):
            errors.append(
                {"field": f"{prefix}.min_ratio", "message": "添加比例必须在 0 到 1 之间"}
            )
        if _finite(il.max_ratio) and not (0.0 <= il.max_ratio <= 1.0):
            errors.append(
                {"field": f"{prefix}.max_ratio", "message": "添加比例必须在 0 到 1 之间"}
            )
        if il.min_ratio > il.max_ratio + 1e-12:
            errors.append(
                {"field": prefix, "message": "最小添加比例不能大于最大添加比例"}
            )

    for i, nb in enumerate(spec.nutrients):
        prefix = f"nutrients[{i}]"
        if nb.code not in known_nutrients:
            errors.append(
                {"field": f"{prefix}.code", "message": f"引用了不存在的营养项: {nb.code}"}
            )
        if nb.lower is None and nb.upper is None:
            errors.append(
                {"field": prefix, "message": "下限和上限至少要提供一个"}
            )
        for name, v in (("lower", nb.lower), ("upper", nb.upper)):
            if v is not None:
                if not _finite(v):
                    errors.append({"field": f"{prefix}.{name}", "message": "必须为有限数"})
                elif v < 0:
                    errors.append(
                        {"field": f"{prefix}.{name}", "message": "营养指标不能为负"}
                    )
        if (
            nb.lower is not None
            and nb.upper is not None
            and nb.lower > nb.upper + 1e-12
        ):
            errors.append({"field": prefix, "message": "下限不能高于上限"})

    seen_ratio: set[tuple[str, str]] = set()
    for i, rc in enumerate(spec.ratios):
        prefix = f"ratios[{i}]"
        if rc.numerator not in known_nutrients:
            errors.append(
                {"field": f"{prefix}.numerator", "message": f"营养项不存在: {rc.numerator}"}
            )
        if rc.denominator not in known_nutrients:
            errors.append(
                {"field": f"{prefix}.denominator", "message": f"营养项不存在: {rc.denominator}"}
            )
        if (rc.numerator, rc.denominator) in seen_ratio:
            errors.append({"field": prefix, "message": "比值约束重复"})
        seen_ratio.add((rc.numerator, rc.denominator))
        if rc.lower is None and rc.upper is None:
            errors.append({"field": prefix, "message": "比值上下限至少提供一个"})
        check_ratio_range(prefix, rc.lower, rc.upper)

    if not _finite(spec.total_mass) or spec.total_mass <= 0:
        errors.append({"field": "total_mass", "message": "总量必须为正数"})

    if errors:
        raise ValidationErrors(errors)
