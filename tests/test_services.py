"""服务层测试：校验、不可行诊断、版本对比、热启动。"""

from __future__ import annotations

import pytest

from tests.conftest import CORN, SBM, WHEAT, recipe_spec

from app.services.validation import ValidationErrors, validate_ingredients, validate_recipe
from app.schemas import (
    IngredientIn,
    IngredientLimitIn,
    NutrientBoundIn,
    RatioIn,
    RecipeIn,
)


def fields_of(exc):
    return {e["field"] for e in exc.errors}


# ------------------------------------------------------------ 数据校验
def test_negative_price_and_nutrient_rejected(db):
    ings = [
        IngredientIn(code="a", name="A", price=-1, nutrients={"DM": 0.9}),
        IngredientIn(code="b", name="B", price=1, nutrients={"DM": -0.5}),
        IngredientIn(code="c", name="C", price=float("nan"), nutrients={"DM": 0.9}),
    ]
    with pytest.raises(ValidationErrors) as ei:
        validate_ingredients(ings, db.nutrient_codes())
    f = fields_of(ei.value)
    assert "ingredients[0].price" in f
    assert "ingredients[1].nutrients.DM" in f
    assert "ingredients[2].price" in f


def test_recipe_bound_validation(db, lib_v1):
    lib = db.get_library()
    base = dict(
        code="r", name="r",
        ingredients=[IngredientLimitIn(code="corn", min_ratio=0.8, max_ratio=0.2)],
        nutrients=[], ratios=[], total_mass=1.0,
    )
    with pytest.raises(ValidationErrors) as ei:
        validate_recipe(RecipeIn(**base), db.nutrient_codes(), set(lib["ingredients"]))
    f = fields_of(ei.value)
    assert "ingredients[0]" in f

    # 比例区间不在 0-1
    base["ingredients"] = [IngredientLimitIn(code="corn", min_ratio=-0.1, max_ratio=1.2)]
    with pytest.raises(ValidationErrors) as ei:
        validate_recipe(RecipeIn(**base), db.nutrient_codes(), set(lib["ingredients"]))
    f = fields_of(ei.value)
    assert "ingredients[0].min_ratio" in f
    assert "ingredients[0].max_ratio" in f

    # 营养上限低于下限 / 引用不存在原料或营养 / 分母全零
    base = dict(
        code="r", name="r",
        ingredients=[IngredientLimitIn(code="corn"), IngredientLimitIn(code="ghost")],
        nutrients=[NutrientBoundIn(code="NOPE", lower=0.5, upper=0.1)],
        ratios=[RatioIn(numerator="CA", denominator="LYS", lower=2, upper=1)],
    )
    with pytest.raises(ValidationErrors) as ei:
        validate_recipe(RecipeIn(**base), db.nutrient_codes(), set(lib["ingredients"]))
    f = fields_of(ei.value)
    assert "ingredients[1].code" in f
    assert "nutrients[0].code" in f
    # 比值分母营养在所有（引用）原料中都为零 —— 在 LP 构建层报
    assert "ratios[0]" in f or any("ratios[0]" in x for x in f)


def test_ratio_zero_denominator_build_error(db, lib_v1):
    # 只用玉米/豆粕，LYS 实际非零；构造一个库里完全没有的营养需先引用合法营养
    # 这里用 ME 作分母不为零的对照，并直接验证全零情形：
    from app.services.optimizer import _to_domain_spec
    from app.solver.lp_builder import build_lp, BuildError, Ingredient
    cats = {
        "corn": Ingredient("corn", "玉米", 0.3, {"CP": 0.08, "ZERO": 0.0}),
        "sbm": Ingredient("sbm", "豆粕", 0.5, {"CP": 0.44, "ZERO": 0.0}),
    }
    spec = _to_domain_spec(recipe_spec(ratios=[("CP", "ZERO", 0.5, None)]))
    with pytest.raises(BuildError) as ei:
        build_lp(spec, cats, {"CP", "ZERO", "DM"})
    assert "denominator" in ei.value.field


# ------------------------------------------------------------ 不可行诊断
def test_infeasible_max_sum_diagnosis(db, lib_v1, opt):
    db.create_recipe(
        "r", "r",
        recipe_spec(bounds={"corn": (0, 0.4), "sbm": (0, 0.4)}),
    )
    rep = opt.optimize("r")
    assert rep["status"] == "infeasible"
    diag = rep["diagnosis"]
    assert diag["reason_code"] == "max_sum_below_total"
    assert "0.8" in diag["message"]
    # 至少给出一组冲突约束
    assert len(diag["iis"]) >= 2


def test_infeasible_nutrient_unattainable(db, lib_v1, opt):
    db.create_recipe(
        "r", "r", recipe_spec(nutrients=(("CP", 0.60, None),))
    )
    rep = opt.optimize("r")
    assert rep["status"] == "infeasible"
    assert rep["diagnosis"]["reason_code"] == "nut_min:CP"
    assert "0.44" in rep["diagnosis"]["message"]


# ------------------------------------------------------------ 即时优化与版本绑定
def test_optimize_binds_versions(db, lib_v1, opt):
    db.create_recipe("r", "r", recipe_spec())
    rep = opt.optimize("r")
    assert rep["status"] == "optimal"
    assert rep["library_version"] == 1
    assert rep["recipe_version"] == 1
    assert rep["certificate"]["optimal_certificate_ok"]
    assert rep["certificate"]["relative_gap"] < 1e-9
    assert set(rep["amounts_kg"]) == {"corn", "sbm"}


def test_version_compare(db, lib_v1, opt):
    db.create_recipe("r", "r", recipe_spec())
    a = opt.optimize("r")
    db.publish_library(
        [{**CORN, "price": 0.34}, {**SBM, "price": 0.54}, WHEAT], "up"
    )
    b = opt.optimize("r")
    cmp_ = opt.compare(a["result_id"], b["result_id"])
    # 组成不变（价格等比变化），成本变化 0.04
    assert cmp_["cost_delta"] == pytest.approx(0.04, abs=1e-9)
    assert all(abs(v) < 1e-9 for v in cmp_["amount_delta"].values())


def test_warm_and_cold_same_cost_service(db, lib_v1, opt):
    db.create_recipe("r", "r", recipe_spec(
        ingredients=("corn", "sbm", "wheat"),
        bounds={"corn": (0.2, 0.8), "sbm": (0.1, 0.6), "wheat": (0.0, 0.5)},
    ))
    cold1 = opt.optimize("r", use_warm=False)
    # 发布新价格（小幅）→ 第二次热启动
    db.publish_library(
        [{**CORN, "price": 0.31}, {**SBM, "price": 0.49},
         {**WHEAT, "price": 0.21}],
        "small shift",
    )
    warm = opt.optimize("r", use_warm=True)
    cold2 = opt.optimize("r", use_warm=False)
    assert warm["warm_start"]["used"] is True
    rel = abs(warm["cost"] - cold2["cost"]) / cold2["cost"]
    assert rel < 1e-9


def test_warm_large_price_change_matches_cold(db, lib_v1, opt):
    """价格剧烈反转导致最优基改变时，热启动经转轴仍应得到与冷启动相同的最优成本。"""

    db.create_recipe("r", "r", recipe_spec(
        ingredients=("corn", "sbm", "wheat"),
        bounds={"corn": (0.0, 1), "sbm": (0.0, 1), "wheat": (0.0, 1)},
        nutrients=(("CP", 0.25, None),),
    ))
    opt.optimize("r", use_warm=False)
    db.publish_library(
        [{**CORN, "price": 9.0}, {**SBM, "price": 0.05},
         {**WHEAT, "price": 7.0}],
        "drastic reversal",
    )
    warm = opt.optimize("r", use_warm=True)
    cold = opt.optimize("r", use_warm=False)
    assert warm["status"] == cold["status"] == "optimal"
    rel = abs(warm["cost"] - cold["cost"]) / max(cold["cost"], 1e-9)
    assert rel < 1e-9


def test_spec_change_falls_back_to_cold(db, lib_v1, opt):
    """约束集变化（structure_key 变）时旧基失效，应回退冷启动且结果正确。"""

    db.create_recipe("r", "r", recipe_spec())
    first = opt.optimize("r")
    # 修改配方：增加麸皮并设置新约束 -> 新版本、新 structure_key
    db.create_recipe(
        "r", "r",
        recipe_spec(
            ingredients=("corn", "sbm", "wheat"),
            bounds={"wheat": (0.05, 0.5)},
            nutrients=(("CP", 0.18, None),),
        ),
        "加麸皮",
    )
    rep = opt.optimize("r")
    assert rep["status"] == "optimal"
    assert rep["certificate"]["optimal_certificate_ok"]
    cold = opt.optimize("r", use_warm=False)
    assert abs(rep["cost"] - cold["cost"]) / cold["cost"] < 1e-9
