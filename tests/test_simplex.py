"""求解器与自证：两原料算例、三条件、对偶间隙、影子价格缩放等。"""

from __future__ import annotations

import numpy as np

from app.solver.certificate import verify
from app.solver.lp_builder import (
    Ingredient,
    IngredientLimit,
    NutrientBound,
    RatioConstraint,
    RecipeSpec,
    build_lp,
)
from app.solver.sensitivity import price_ranges
from app.solver.simplex import Simplex

CORN_ING = Ingredient("corn", "玉米", 0.30, {"CP": 0.08})
SBM_ING = Ingredient("sbm", "豆粕", 0.50, {"CP": 0.44})


def two_ing(prices=(0.30, 0.50)):
    corn = Ingredient("corn", "玉米", prices[0], {"CP": 0.08})
    sbm = Ingredient("sbm", "豆粕", prices[1], {"CP": 0.44})
    spec = RecipeSpec(
        "r1", "两原料",
        (IngredientLimit("corn", 0, 1), IngredientLimit("sbm", 0, 1)),
        (NutrientBound("CP", lower=0.18),),
    )
    built = build_lp(spec, {"corn": corn, "sbm": sbm}, {"CP", "DM"})
    return spec, built


# ------------------------------------------------------------ 手算核对算例
def test_two_ingredient_hand_calc():
    import pytest
    _, built = two_ing()
    res = Simplex(built.costs, built.rows).solve()
    assert res.status == "optimal"
    assert res.x[0] == pytest.approx(0.72222222, abs=1e-6)
    assert res.x[1] == pytest.approx(0.27777778, abs=1e-6)
    assert res.objective == pytest.approx(0.3555555556, abs=1e-9)
    # 豆粕约 0.2778、玉米约 0.7222、成本约 0.3556
    assert abs(res.x[1] - 0.2778) < 1e-4
    assert abs(res.x[0] - 0.7222) < 1e-4


def test_two_ingredient_pytest_approx():
    _, built = two_ing()
    res = Simplex(built.costs, built.rows).solve()
    import pytest
    assert res.x[0] == pytest.approx(0.72222222, abs=1e-6)
    assert res.x[1] == pytest.approx(0.27777778, abs=1e-6)
    assert res.objective == pytest.approx(0.3555555556, abs=1e-9)


# ------------------------------------------------------------ 最优性三条件
def test_optimality_certificate():
    _, built = two_ing()
    res = Simplex(built.costs, built.rows).solve()
    cert = verify(built, res)
    assert cert["primal_feasible"]
    assert cert["dual_feasible"]
    assert cert["complementary_slackness"]
    assert cert["relative_gap"] < 1e-9
    # 逐条代入：CP 行起作用，影子价格 0.5556；总量行 0.2556
    by_key = {r["key"]: r for r in cert["rows"]}
    assert by_key["nut_min:CP"]["binding"]
    assert by_key["nut_min:CP"]["dual"] > 0
    assert by_key["total"]["binding"]
    # 非负用量、互补松弛逐条
    assert all(c["feasible_ok"] and c["complementary_ok"] for c in cert["columns"])


def test_shadow_price_relaxation():
    """放宽一条起作用的下限约束，最优成本不升；收紧则不降。"""

    def cost_with_cp(cp_lower):
        spec = RecipeSpec(
            "r", "r",
            (IngredientLimit("corn"), IngredientLimit("sbm")),
            (NutrientBound("CP", lower=cp_lower),),
        )
        b = build_lp(spec, {"corn": CORN_ING, "sbm": SBM_ING}, {"CP", "DM"})
        return Simplex(b.costs, b.rows).solve().objective

    base = cost_with_cp(0.18)
    relaxed = cost_with_cp(0.16)
    tightened = cost_with_cp(0.20)
    assert relaxed <= base + 1e-12
    assert tightened >= base - 1e-12
    # 影子价格近似等于边际变化（下降量）
    assert abs((base - relaxed) / 0.02 - 0.5555556) < 1e-6


# ------------------------------------------- 单价同乘正数：用量不变，同比放大
def test_price_scaling_invariance():
    _, b1 = two_ing((0.30, 0.50))
    k = 2.7
    _, b2 = two_ing((0.30 * k, 0.50 * k))
    r1 = Simplex(b1.costs, b1.rows).solve()
    r2 = Simplex(b2.costs, b2.rows).solve()
    assert np.allclose(r1.x, r2.x, atol=1e-10)
    assert abs(r2.objective - k * r1.objective) < 1e-9
    for key in r1.duals:
        assert abs(r2.duals[key] - k * r1.duals[key]) < 1e-9


# ------------------------------------------------------------ 灵敏度区间
def test_sensitivity_ranges():
    import pytest
    _, built = two_ing()
    res = Simplex(built.costs, built.rows).solve()
    ranges = price_ranges(built, res)
    # 玉米 <=0.5（超过豆粕价）、豆粕 >=0.3
    assert ranges["corn"]["upper_price"] == pytest.approx(0.5, abs=1e-9)
    assert ranges["sbm"]["lower_price"] == pytest.approx(0.3, abs=1e-9)

    def solve(prices):
        _, b = two_ing(prices)
        return Simplex(b.costs, b.rows).solve().x

    base = solve((0.30, 0.50))
    # 区间内：组成不变
    assert np.allclose(solve((0.45, 0.50)), base, atol=1e-9)
    assert np.allclose(solve((0.30, 0.40)), base, atol=1e-9)
    # 出区间：组成改变
    assert not np.allclose(solve((0.55, 0.50)), base, atol=1e-9)
    assert not np.allclose(solve((0.30, 0.25)), base, atol=1e-9)


# ------------------------------------------------------------ 退化与确定性
def test_degeneracy_determinism():
    """高度约束/多次连解：结果必须逐位一致。"""

    specs = []
    codes = ["corn", "sbm", "wheat"]
    A = np.array([
        [0.08, 0.44, 0.14],
        [0.001, 0.003, 0.001],
        [0.003, 0.006, 0.009],
    ])
    cats = {
        "corn": Ingredient("corn", "玉米", 0.30, {"CP": 0.08, "CA": 0.001, "TP": 0.003}),
        "sbm": Ingredient("sbm", "豆粕", 0.50, {"CP": 0.44, "CA": 0.003, "TP": 0.006}),
        "wheat": Ingredient("wheat", "麸皮", 0.20, {"CP": 0.14, "CA": 0.001, "TP": 0.009}),
    }
    spec = RecipeSpec(
        "r", "退化",
        tuple(IngredientLimit(c, 0.0, 0.9) for c in codes),
        (NutrientBound("CP", lower=0.16), NutrientBound("CA", lower=0.0015)),
        (RatioConstraint("CA", "TP", 0.3, 1.0),),
    )
    built = build_lp(spec, cats, {"CP", "CA", "TP", "DM"})
    results = [Simplex(built.costs, built.rows).solve() for _ in range(6)]
    x0 = results[0].x
    c0 = results[0].objective
    for r in results[1:]:
        assert np.array_equal(r.x, x0)
        assert r.objective == c0
    cert = verify(built, results[0])
    assert cert["optimal_certificate_ok"]


# ------------------------------------------------------------ 比值约束
def test_ratio_constraint():
    import pytest
    cats = {
        "corn": Ingredient("corn", "玉米", 0.30, {"CA": 0.001, "TP": 0.003}),
        "sbm": Ingredient("sbm", "豆粕", 0.50, {"CA": 0.003, "TP": 0.006}),
        "stone": Ingredient("stone", "石粉", 0.10, {"CA": 0.38, "TP": 0.0}),
    }
    spec = RecipeSpec(
        "r", "钙磷比",
        tuple(IngredientLimit(c) for c in cats),
        (),
        (RatioConstraint("CA", "TP", 1.2, 2.0),),
    )
    built = build_lp(spec, cats, {"CA", "TP", "DM"})
    res = Simplex(built.costs, built.rows).solve()
    assert res.status == "optimal"
    ca = sum(cats[c].nutrients["CA"] * res.x[j] for j, c in enumerate(built.ingredients))
    tp = sum(cats[c].nutrients["TP"] * res.x[j] for j, c in enumerate(built.ingredients))
    assert 1.2 - 1e-7 <= ca / tp <= 2.0 + 1e-7
    cert = verify(built, res)
    assert cert["optimal_certificate_ok"]


# ------------------------------------------------------------ 热启动一致性
def test_warm_equals_cold():
    _, built = two_ing()
    cold = Simplex(built.costs, built.rows).solve()
    warm = Simplex(built.costs, built.rows).solve(warm_basis=cold.basis)
    assert warm.status == "optimal"
    assert np.allclose(warm.x, cold.x, atol=1e-10)
    assert abs(warm.objective - cold.objective) / cold.objective < 1e-9


def test_warm_after_price_change():
    _, b1 = two_ing((0.30, 0.50))
    r1 = Simplex(b1.costs, b1.rows).solve()
    _, b2 = two_ing((0.33, 0.48))
    warm = Simplex(b2.costs, b2.rows).solve(warm_basis=r1.basis)
    cold = Simplex(b2.costs, b2.rows).solve()
    assert abs(warm.objective - cold.objective) / cold.objective < 1e-9
    assert np.allclose(warm.x, cold.x, atol=1e-9)
