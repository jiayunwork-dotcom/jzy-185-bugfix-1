"""价格灵敏度区间：在该区间内变化时最优基不变，因而原料组成不变。

设终态表格目标行为检验数行（c-z 约定、入基看负、RHS=-z）。
令原料列 j 在表格中的系数为列 ``d``（含目标行系数 rc_j）。

* 若 j 是基变量（在第 r 行出基），其价格变化 Δc_j 通过行变换影响所有非基列：
      rc_k(Δ) = rc_k(0) - Δc_j * ā_{r,k}，要求 rc_k(Δ) >= 0
  - ā_{r,k} > 0 给上限 Δ <= rc_k/ā
  - ā_{r,k} < 0 给下限 Δ >= rc_k/ā
* 若 j 是非基变量，则只影响它自己：rc_j(Δ) = rc_j(0) + Δc_j >= 0，
  即 Δc_j >= -rc_j(0)（上限 +∞）。

注意：我们只对“非人工列”（原料列、slack、surplus）做比值，因为人工列在
第二阶段被禁止入基，不能限制区间。
"""

from __future__ import annotations

import math

import numpy as np

from .lp_builder import BuiltLP
from .simplex import EPS, LPResult

INF = math.inf


def price_ranges(built: BuiltLP, res: LPResult) -> dict[str, dict]:
    assert res.tableau is not None and res.basis is not None
    tab = res.tableau
    basis = res.basis
    n = len(built.ingredients)

    # 非人工、非基的列下标
    allowed = [
        j
        for j in range(tab.shape[1] - 1)
        if res.columns is not None and res.columns[j][0] != "art"
    ]
    nonbasic = [j for j in allowed if j not in basis]

    base_row_of: dict[int, int] = {b: r for r, b in enumerate(basis)}

    out: dict[str, dict] = {}
    for j, code in enumerate(built.ingredients):
        price = float(built.costs[j])
        if j in base_row_of:
            r = base_row_of[j]
            lower_delta = -INF
            upper_delta = INF
            blockers_lo: list[str] = []
            blockers_hi: list[str] = []
            for k in nonbasic:
                a = tab[r, k]
                rc = tab[-1, k]
                if abs(a) <= EPS:
                    continue
                ratio = rc / a
                if a > EPS:
                    # Δ <= rc/ā（上限）
                    if ratio < upper_delta:
                        upper_delta = ratio
                        blockers_hi = [_col_name(k, built, res)]
                    elif abs(ratio - upper_delta) <= 1e-10:
                        blockers_hi.append(_col_name(k, built, res))
                else:
                    # Δ >= rc/ā（下限，ā<0）
                    if ratio > lower_delta:
                        lower_delta = ratio
                        blockers_lo = [_col_name(k, built, res)]
                    elif abs(ratio - lower_delta) <= 1e-10:
                        blockers_lo.append(_col_name(k, built, res))
            low = price + lower_delta if lower_delta > -INF else -INF
            high = price + upper_delta if upper_delta < INF else INF
            out[code] = {
                "price": price,
                "lower_price": _clean(low),
                "upper_price": _clean(high),
                "lower_blocking": blockers_lo,
                "upper_blocking": blockers_hi,
                "basic": True,
            }
        else:
            rc0 = float(tab[-1, j])
            out[code] = {
                "price": price,
                "lower_price": _clean(price - rc0),
                "upper_price": None,  # +∞
                "lower_blocking": [code],
                "upper_blocking": [],
                "basic": False,
            }
    return out


def _col_name(k: int, built: BuiltLP, res: LPResult) -> str:
    assert res.columns is not None
    kind, ing, row_key = res.columns[k]
    if kind == "ing" and ing is not None:
        return built.ingredients[ing]
    meta = built.constraint_info.get(row_key or "", {})
    return meta.get("label", row_key or f"col{k}")


def _clean(v: float) -> float:
    return None if (v == INF or v == -INF) else float(v)
