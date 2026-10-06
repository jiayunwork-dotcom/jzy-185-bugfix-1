"""最优性证书：把原始解与对偶解逐条代入检验。

三项条件（对最小化 LP ``min c^T x, A x <op> b, x>=0``）
------------------------------------------------------
1. 原始可行：每条约束的残差方向正确、x>=0；
2. 对偶可行：由行类型决定的对偶符号正确
   （<= 行 y<=0，>= 行 y>=0，= 行 y 自由），且每个原料检验数
   ``rc_j = c_j - y^T A_j >= 0``；
3. 互补松弛：
   - 原始行不起作用（有正松弛/剩余）时 y=0；起作用时 y 可非零；
   - 原料 x_j>0 时检验数为 0；rc_j>0 时 x_j=0。

同时计算原始目标与对偶目标的相对差（应在 1e-9 内）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lp_builder import BuiltLP
from .simplex import LPResult

CHECK_TOL = 1e-8
GAP_TOL = 1e-9


@dataclass
class RowCheck:
    key: str
    kind: str
    label: str
    sense: str
    lhs: float
    rhs: float
    slack: float          # 对 <=: rhs-lhs；对 >=: lhs-rhs；=: -|lhs-rhs|
    binding: bool
    dual: float
    dual_ok: bool         # 对偶符号正确
    complementary_ok: bool
    detail: str


@dataclass
class ColumnCheck:
    ingredient: str
    amount: float
    price: float
    reduced_cost: float
    feasible_ok: bool     # x>=0
    complementary_ok: bool  # x>0 => rc==0；rc>0 => x==0


def verify(built: BuiltLP, res: LPResult, tol: float = CHECK_TOL) -> dict:
    assert res.x is not None and res.reduced_costs is not None
    x = res.x
    row_checks: list[RowCheck] = []

    primal_ok = bool(np.all(x >= -tol))
    dual_ok_all = True
    comp_ok_all = True

    for row in built.rows:
        lhs = float(np.dot(row.a, x))
        y = res.duals.get(row.key, 0.0)
        if row.sense == "<=":
            slack = row.b - lhs
            binding = abs(slack) <= tol
            dual_ok = y <= tol
        elif row.sense == ">=":
            slack = lhs - row.b
            binding = abs(slack) <= tol
            dual_ok = y >= -tol
        else:
            slack = -abs(lhs - row.b)
            binding = abs(lhs - row.b) <= tol
            dual_ok = True
        # 互补松弛：非起作用行的对偶必须为 0
        comp_ok = binding or abs(y) <= tol
        row_ok = slack >= -tol
        meta = built.constraint_info.get(row.key, {})
        primal_ok &= row_ok
        dual_ok_all &= dual_ok
        comp_ok_all &= comp_ok
        detail = (
            ("起作用(binding)" if binding else f"松弛 {slack:.3g}")
            + f"，对偶价格 {y:.6g}"
        )
        row_checks.append(
            RowCheck(
                key=row.key,
                kind=meta.get("kind", row.key),
                label=meta.get("label", row.key),
                sense=row.sense,
                lhs=lhs,
                rhs=row.b,
                slack=slack,
                binding=binding,
                dual=y,
                dual_ok=dual_ok,
                complementary_ok=comp_ok,
                detail=detail,
            )
        )

    col_checks: list[ColumnCheck] = []
    for j, code in enumerate(built.ingredients):
        xj, rcj = float(x[j]), float(res.reduced_costs[j])
        feas = xj >= -tol
        comp = (abs(xj) <= tol) or (abs(rcj) <= tol)
        if rcj > tol and xj > tol:
            comp = False
        primal_ok &= feas
        dual_ok_all &= rcj >= -tol
        comp_ok_all &= comp
        col_checks.append(
            ColumnCheck(
                ingredient=code,
                amount=xj,
                price=float(built.costs[j]),
                reduced_cost=rcj,
                feasible_ok=feas,
                complementary_ok=comp,
            )
        )

    primal_obj = float(np.dot(built.costs, x))
    # 对偶目标 y^T b：注意 >= / = 行的 b 用原值（求解器内部已统一符号）
    dual_obj = 0.0
    for row in built.rows:
        dual_obj += res.duals.get(row.key, 0.0) * row.b
    denom = max(abs(primal_obj), abs(dual_obj), 1e-12)
    rel_gap = abs(primal_obj - dual_obj) / denom

    return {
        "primal_objective": primal_obj,
        "dual_objective": dual_obj,
        "absolute_gap": abs(primal_obj - dual_obj),
        "relative_gap": rel_gap,
        "primal_feasible": primal_ok,
        "dual_feasible": dual_ok_all,
        "complementary_slackness": comp_ok_all,
        "optimal_certificate_ok": primal_ok
        and dual_ok_all
        and comp_ok_all
        and rel_gap <= GAP_TOL,
        "rows": [c.__dict__ for c in row_checks],
        "columns": [c.__dict__ for c in col_checks],
    }
