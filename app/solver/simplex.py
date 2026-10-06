"""从零实现的两阶段单纯形法。

不依赖任何线性规划/优化第三方库，只用 NumPy 做数组运算。

设计要点
--------
* 标准型：``min c^T x``，``A x (op) b``，原料用量 x >= 0。
  ``<=`` 行加松弛变量(slack)，``>=`` 行加剩余变量(surplus)+人工变量，
  ``=`` 行加人工变量。
* 每个原料的 ``x_i >= l_i`` 通过一条显式 ``>=`` 行表达（仅当 l_i>0），
  ``x_i <= u_i`` 通过一条显式 ``<=`` 行表达（仅当 u_i<1）。
  因此所有变量保持自由非负形式，检验数、互补松弛、灵敏度的规则统一。
* 第一阶段最小化人工变量之和；第二阶段用真实成本重建目标行。
* 退化/循环：全程使用 Bland 规则（按最小下标选入基、按最小比值平局离开基），
  数值阈值统一为 EPS。同样的输入必然得到同样的解。
* 表格中保留所有辅助列：slack/surplus/人工各占一列，第二阶段人工列系数恒为 0
  且禁止入基。这样对偶变量可直接从目标行对应辅助列读出，无需重新解方程组。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# 转轴运算的数值阈值：小于它视为 0
EPS = 1e-11
# 可行性判定允许的误差
FEAS_EPS = 1e-9

Sense = Literal["<=", ">=", "="]


@dataclass
class Row:
    """一条原始约束行 a @ x <sense> b。"""

    key: str
    a: np.ndarray
    sense: Sense
    b: float
    meta: dict = field(default_factory=dict)


@dataclass
class LPResult:
    status: Literal["optimal", "infeasible", "unbounded"]
    x: np.ndarray | None = None
    objective: float = 0.0
    # 每条原始行（按 rows 顺序）的对偶乘子；被删除的冗余行给 0
    duals: dict[str, float] = field(default_factory=dict)
    reduced_costs: np.ndarray | None = None
    # 最优基：长度为存活行数，元素为基变量的列下标
    basis: list[int] | None = None
    # 终态完整表格（含目标行）与存活行对应的原始行下标，供灵敏度复用
    tableau: np.ndarray | None = None
    alive_row_index: list[int] | None = None
    # 列信息：(类型, 原料列下标或None, 行key)
    columns: list[tuple[str, int | None, str | None]] | None = None
    first_phase_infeas: float = 0.0


def _norm_rows(rows: list[Row]) -> list[Row]:
    """把 b<0 的行翻转，使右端项非负（给人工变量一个合法初始基）。"""

    flip = {"<=": ">=", ">=": "<=", "=": "="}
    out: list[Row] = []
    for r in rows:
        b = float(r.b)
        a = np.array(r.a, dtype=float)
        sense = r.sense
        if b < -EPS:
            a, b, sense = -a, -b, flip[sense]
        out.append(Row(r.key, a, sense, b, dict(r.meta)))
    return out


class Simplex:
    """对一份列顺序已固定的 LP 做两阶段求解。

    表格 body 的行序始终与 ``self.alive``（存活行的原始下标）一一对应；
    删除冗余行时两者同步删除。
    """

    def __init__(self, c, rows: list[Row]):
        self.c = np.asarray(c, dtype=float)
        # 先规范化（翻转负右端项），列结构基于规范化后的行构建
        self.rows = _norm_rows(rows)
        self.n = self.c.shape[0]
        self.m = len(self.rows)
        # 列信息：(kind, ingredient_col, row_key)
        self.cols: list[tuple[str, int | None, str | None]] = [
            ("ing", j, None) for j in range(self.n)
        ]
        self.slack_col: dict[int, int] = {}
        self.surplus_col: dict[int, int] = {}
        self.art_col_of: dict[int, int] = {}  # 行下标 -> 人工列
        self.tableau = self._build()
        self.ncols_total = self.tableau.shape[1] - 1
        self.alive: list[int] = list(range(self.m))

    # ------------------------------------------------------------------ build
    def _build(self) -> np.ndarray:
        blocks = [np.zeros((self.m, self.n))]
        for i, r in enumerate(self.rows):
            blocks[0][i] = r.a
        for i, r in enumerate(self.rows):
            offset = sum(b.shape[1] for b in blocks)  # 不含本块前的列数
            if r.sense == "<=":
                col = np.zeros((self.m, 1))
                col[i, 0] = 1.0
                blocks.append(col)
                self.cols.append(("slack", None, r.key))
                self.slack_col[i] = offset
            elif r.sense == ">=":
                sur = np.zeros((self.m, 1))
                sur[i, 0] = -1.0
                art = np.zeros((self.m, 1))
                art[i, 0] = 1.0
                blocks.extend([sur, art])
                self.cols.append(("surplus", None, r.key))
                self.surplus_col[i] = offset
                self.cols.append(("art", None, r.key))
                self.art_col_of[i] = offset + 1
            else:  # '='
                art = np.zeros((self.m, 1))
                art[i, 0] = 1.0
                blocks.append(art)
                self.cols.append(("art", None, r.key))
                self.art_col_of[i] = offset
        b = np.array([r.b for r in self.rows], dtype=float).reshape(self.m, 1)
        return np.hstack(blocks + [b])

    def _initial_basis(self) -> list[int]:
        """每一行的初始基：slack 行用 slack 列，其余用人工列。"""

        basis: list[int] = [-1] * self.m
        for ri, ci in self.slack_col.items():
            basis[ri] = ci
        for ri, ci in self.art_col_of.items():
            basis[ri] = ci
        return basis

    # ------------------------------------------------------------- pivoting
    @staticmethod
    def _pivot(tab: np.ndarray, r: int, ent: int):
        """以 tab[r, ent] 为主元做 Gauss-Jordan 消元（含目标行）。"""

        tab[r] /= tab[r, ent]
        for i in range(tab.shape[0]):
            if i == r:
                continue
            factor = tab[i, ent]
            if abs(factor) > 0.0:
                tab[i] -= factor * tab[r]

    def _run_phase(
        self, tab: np.ndarray, basis: list[int], allow_cols: set[int] | None
    ) -> str:
        """Bland 单纯形主循环。表格行序 = self.alive 序，目标行为最后一行。

        目标行约定（最小化标准单纯形）：系数为检验数 c_j - z_j，
        严格为负的列入基；RHS 为 -z。
        """

        while True:
            obj = tab[-1]
            ent = -1
            for j in range(self.ncols_total):
                if allow_cols is not None and j not in allow_cols:
                    continue
                if obj[j] < -EPS:
                    ent = j
                    break
            if ent < 0:
                return "optimal"
            best_order = -1
            best_ratio = np.inf
            for order in range(len(self.alive)):
                a = tab[order, ent]
                if a > EPS:
                    ratio = tab[order, -1] / a
                    if ratio < best_ratio - EPS:
                        best_ratio, best_order = ratio, order
                    elif abs(ratio - best_ratio) <= EPS and (
                        best_order < 0 or order < best_order
                    ):
                        # Bland 平局：取行序最小（alive 序即字典序）
                        best_order = order
            if best_order < 0:
                return "unbounded"
            self._pivot(tab, best_order, ent)
            basis[best_order] = ent

    # ------------------------------------------------------------- phase 1
    def _phase1_objective(self, body: np.ndarray) -> np.ndarray:
        """第一阶段目标行（检验数 c-z：负值入基；RHS = -w）。

        人工变量成本为 +1：初始行在人工列放 +1，再减去每个基人工行，
        使基人工列归零，此时原料列为 -sum(a)、RHS 为 -sum(b)。
        """

        row = np.zeros(self.ncols_total + 1)
        for ci in self.art_col_of.values():
            row[ci] = 1.0
        for ri in self.art_col_of:
            row -= body[ri]
        for ci in self.art_col_of.values():
            row[ci] = 0.0
        return row

    # ---------------------------------------------------------------- solve
    def solve(self, warm_basis: list[int] | None = None) -> LPResult:
        if self.m == 0:
            if np.any(self.c < -EPS):
                return LPResult("unbounded")
            return LPResult(
                "optimal",
                x=np.zeros(self.n),
                objective=0.0,
                reduced_costs=self.c.copy(),
                basis=[],
                tableau=None,
                alive_row_index=[],
                columns=list(self.cols),
            )

        # ---------------- 热启动 ----------------
        if warm_basis is not None and len(warm_basis) == self.m and len(
            set(warm_basis)
        ) == len(warm_basis) and all(0 <= b_ <= self.ncols_total - 1 for b_ in warm_basis):
            tab = self._canonical_from_basis(warm_basis)
            if tab is not None and np.all(tab[:-1, -1] >= -FEAS_EPS):
                basis = list(warm_basis)
                status = self._run_phase(tab, basis, self._non_art_cols())
                if status != "unbounded":
                    return self._collect(tab, basis)
            # 基失效：回退到两阶段冷启动
            self.alive = list(range(self.m))

        # ---------------- 冷启动：第一阶段 ----------------
        body = self.tableau.copy()
        basis = self._initial_basis()
        tab = np.vstack([body, self._phase1_objective(body)])
        status = self._run_phase(tab, basis, None)
        infeas_val = -tab[-1, -1]  # RHS = -w（w 为人工变量之和）
        if infeas_val > FEAS_EPS:
            return LPResult("infeasible", first_phase_infeas=float(infeas_val))

        # 转出/删除残留人工变量（冗余行）
        art_set = set(self.art_col_of.values())
        while True:
            target = -1
            for order, bi in enumerate(basis):
                if bi in art_set:
                    target = order
                    break
            if target < 0:
                break
            row = tab[target]
            pivot_col = -1
            for j in range(self.ncols_total):
                if j in art_set:
                    continue
                if abs(row[j]) > EPS:
                    pivot_col = j
                    break
            if pivot_col >= 0:
                self._pivot(tab, target, pivot_col)
                basis[target] = pivot_col
                continue
            # 冗余行：同步删除表格约束行与 alive
            ri = self.alive[target]
            keep = [k for k in range(len(self.alive)) if k != target]
            tab = tab[np.array(keep + [len(self.alive)], dtype=int)]
            self.alive.remove(ri)
            basis.pop(target)
            if not self.alive:
                break

        # ---------------- 第二阶段 ----------------
        # 目标行初始为 [c | 0]（检验数 c-z 约定、RHS=-z），再按当前基做行变换，
        # 使基变量列检验数归零。
        tab[-1].fill(0.0)
        tab[-1, : self.n] = self.c
        for order, bi in enumerate(basis):
            if abs(tab[-1, bi]) > EPS:
                tab[-1] -= tab[-1, bi] * tab[order]
        # 人工列禁止入基（由 allow_cols 保证），但其检验数要保留：
        # 等式行的对偶乘子 y 就是从人工列读出的，不能清零。
        status = self._run_phase(tab, basis, self._non_art_cols())
        if status == "unbounded":
            return LPResult("unbounded")
        return self._collect(tab, basis)

    def _non_art_cols(self) -> set[int]:
        return {j for j in range(self.ncols_total) if self.cols[j][0] != "art"}

    def _canonical_from_basis(self, warm_basis: list[int]) -> np.ndarray | None:
        """由给定基重建规范型表格（含规范化的真实目标行）。奇异则返回 None。"""

        body = self.tableau.copy()
        B = body[:, np.array(warm_basis)]
        try:
            canon = np.linalg.solve(B, body)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(canon)):
            return None
        unit = canon[:, np.array(warm_basis)]
        if np.max(np.abs(unit - np.eye(self.m))) > 1e-7:
            return None
        obj = np.zeros(self.ncols_total + 1)
        obj[: self.n] = self.c
        full = np.vstack([canon, obj[np.newaxis, :]])
        for ri, bi in enumerate(warm_basis):
            if abs(full[-1, bi]) > EPS:
                full[-1] -= full[-1, bi] * full[ri]
        # 人工列检验数保留（等式行对偶需要），仅靠 allow_cols 禁止其入基
        return full

    # ------------------------------------------------------------- output
    def _collect(self, tab: np.ndarray, basis: list[int]) -> LPResult:
        x = np.zeros(self.n)
        for order, bi in enumerate(basis):
            kind, ing, _ = self.cols[bi]
            if kind == "ing":
                assert ing is not None
                x[ing] = tab[order, -1]
        duals: dict[str, float] = {}
        for i, r in enumerate(self.rows):
            if i not in self.alive:
                duals[r.key] = 0.0
        duals: dict[str, float] = {}
        for i, r in enumerate(self.rows):
            if i not in self.alive:
                duals[r.key] = 0.0
            elif i in self.slack_col:
                # <= 行（上限类）：slack 列检验数 = -y
                duals[r.key] = float(-tab[-1, self.slack_col[i]])
            elif i in self.surplus_col:
                # >= 行（下限类）：surplus 列检验数 = y
                duals[r.key] = float(tab[-1, self.surplus_col[i]])
            else:
                # 等式行：人工列检验数 = -y
                ci = self.art_col_of.get(i)
                duals[r.key] = float(-tab[-1, ci]) if ci is not None else 0.0
        return LPResult(
            status="optimal",
            x=x,
            objective=float(-tab[-1, -1]),  # 目标行 RHS = -z
            duals=duals,
            # 原料列检验数 c_j - z_j：最优时非基原料 >= 0，基原料 = 0
            reduced_costs=np.array(tab[-1, : self.n], dtype=float),
            basis=list(basis),
            tableau=tab.copy(),
            alive_row_index=list(self.alive),
            columns=list(self.cols),
        )
