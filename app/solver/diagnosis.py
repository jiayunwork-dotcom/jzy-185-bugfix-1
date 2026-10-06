"""不可行诊断：不能只回“无解”，要给出至少一组能解释冲突的约束。

两层策略
--------
1. 快速结构性检查（配方师最常见的错误，解释也最直观）：
   a. 原料最大添加比例之和 < 总量（上限加起来不足 1）；
   b. 某营养下限高于所有原料在各自上限内能达到的最大含量；
   c. 某营养上限低于所有原料在各自下限内必须达到的最小含量；
   d. 比值约束：即使取“最有利配比”，仍无法达到下限/上限。
2. 若结构检查没命中，用删除过滤法（deletion filter）求一组极小不可行约束子集
   (IIS-lite)：逐条尝试删除约束，删后仍不可行就真删，删后可行就保留，
   最终留下的集合本身不可行、且去掉任一条就可行——它就是冲突解释。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lp_builder import BuiltLP
from .simplex import FEAS_EPS, Simplex


@dataclass
class Conflict:
    code: str
    label: str
    detail: str


def _infeasible(rows) -> bool:
    if not rows:
        return False
    res = Simplex(np.zeros(len(rows[0].a)), rows).solve()
    return res.status == "infeasible"


def diagnose(built: BuiltLP) -> dict:
    conflicts: list[Conflict] = []
    info = built.constraint_info

    # 先从行中还原每种原料的添加区间（缺行时下限 0、上限总量）
    total_mass = next(
        r.b for r in built.rows if info.get(r.key, {}).get("kind") == "total"
    )
    n = len(built.ingredients)
    umax = np.full(n, total_mass)
    lmin = np.zeros(n)
    code_to_j = {c: j for j, c in enumerate(built.ingredients)}
    for r in built.rows:
        m = info.get(r.key, {})
        if m.get("kind") == "ingredient_max":
            umax[code_to_j[m["ingredient"]]] = r.b
        elif m.get("kind") == "ingredient_min":
            lmin[code_to_j[m["ingredient"]]] = r.b

    # a) 添加上限之和 < 总量
    max_sum = float(umax.sum())
    if max_sum < total_mass - 1e-9:
        conflicts.append(
            Conflict(
                "max_sum_below_total",
                "各原料最大添加比例之和小于总量",
                f"上限合计 {max_sum:.6g} < 总量 {total_mass:g}，无论怎么配都凑不够总量",
            )
        )
        for code, j in code_to_j.items():
            conflicts.append(
                Conflict(
                    f"ing_max:{code}",
                    f"{code} 最大添加 {umax[j]:g}",
                    "参与该上限和",
                )
            )

    # b/c) 营养可达性
    N = built.nutrient_matrix
    assert N is not None
    total_hi = max_sum

    for r_ in built.rows:
        m = info.get(r_.key, {})
        if m.get("kind") == "nutrient_min":
            code = m["nutrient"]
            row = N[built.nutrient_codes.index(code)]
            # 最有利：尽量多用高含量原料（0-1 贪心，单约束松弛问题）
            order = np.argsort(-row)
            remain = total_mass
            attain = 0.0
            caps = umax.copy()
            for j in order:
                take = min(caps[j], remain)
                attain += take * row[j]
                remain -= take
            if attain + 1e-9 < r_.b:
                conflicts.append(
                    Conflict(
                        r_.key,
                        m["label"],
                        f"{code} 下限 {r_.b:g} 高于各原料在上限内能达到的最大值 "
                        f"{attain:.6g}",
                    )
                )
        elif m.get("kind") == "nutrient_max":
            code = m["nutrient"]
            row = N[built.nutrient_codes.index(code)]
            # 最不利：最小必须用量带来的最低含量（总量没填满的部分可填最低含量原料）
            forced = sum(lmin[j] for j in range(n))
            attain = float(np.dot(row, lmin))
            remain = total_mass - forced
            j_fill = int(np.argmin(row))
            # 被最小用量占掉的不能再填；简化：把剩余量给含量最低且有容量的原料
            free_caps = umax - lmin
            order = np.argsort(row)
            for j in order:
                take = min(free_caps[j], remain)
                attain += take * row[j]
                remain -= take
            if attain - 1e-9 > r_.b:
                conflicts.append(
                    Conflict(
                        r_.key,
                        m["label"],
                        f"{code} 上限 {r_.b:g} 低于各原料在下限内必须达到的最小值 "
                        f"{attain:.6g}",
                    )
                )

    # d) 比值可达性：用比值线性化行检查“最有利”方向
    for key in built.ratio_rows:
        m = info[key]
        r_ = next(x for x in built.rows if x.key == key)
        # 忽略总量/其他约束的松弛：求 max(a·x) 或 min(a·x) 在添加上限内
        row = r_.a
        if m["kind"] == "ratio_lower":
            best = _extreme_attain(row, umax, total_mass, maximize=True)
            if best < -1e-9:
                conflicts.append(
                    Conflict(
                        key,
                        m["label"],
                        f"比值 {m['numerator']}/{m['denominator']} 即使取最有利配比，"
                        f"线性化值最大仅 {best:.6g}（需 ≥0），下限 {m['bound']:g} 无法达到",
                    )
                )
        else:
            best = _extreme_attain(row, umax, total_mass, maximize=False)
            if best > 1e-9:
                conflicts.append(
                    Conflict(
                        key,
                        m["label"],
                        f"比值 {m['numerator']}/{m['denominator']} 即使取最有利配比，"
                        f"线性化值最小仍为 {best:.6g}（需 ≤0），上限 {m['bound']:g} 无法达到",
                    )
                )

    # ---------- IIS-lite 兜底/强化 ----------
    iis_rows = _deletion_filter(built.rows)
    iis = [
        {
            "key": r.key,
            "label": info.get(r.key, {}).get("label", r.key),
            "kind": info.get(r.key, {}).get("kind", ""),
        }
        for r in iis_rows
    ]

    if conflicts:
        primary = conflicts[0]
        return {
            "reason_code": primary.code,
            "message": primary.detail,
            "conflicts": [c.__dict__ for c in conflicts],
            "iis": iis,
        }

    return {
        "reason_code": "iis",
        "message": "以下约束集合互相冲突（去掉其中任意一条即可行）："
        + "；".join(x["label"] for x in iis),
        "conflicts": [
            Conflict(x["key"], x["label"], "属于极小不可行约束子集").__dict__
            for x in iis
        ],
        "iis": iis,
    }


def _extreme_attain(row: np.ndarray, caps: np.ndarray, total: float, maximize: bool):
    """忽略营养约束，仅在 0<=x<=caps、sum x=total 下求 row·x 的最大/最小值。"""

    order = np.argsort(-row if maximize else row)
    remain = total
    val = 0.0
    for j in order:
        take = min(caps[j], remain)
        val += take * row[j]
        remain -= take
    return val


def _deletion_filter(rows) -> list:
    """删除过滤：返回一组极小不可行行子集。"""

    working = list(rows)
    if not _infeasible(working):
        return []
    kept: list = []
    for r in rows:
        trial = [x for x in working if x is not r]
        if _infeasible(trial):
            working = trial  # 该约束不是必需，删除
        else:
            kept.append(r)
    return kept
