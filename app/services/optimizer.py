"""优化服务：组装 LP、调用自研单纯形、产出带自证的报告并持久化。

热启动策略（文档亦同）
----------------------
每周换价只改价格、不改结构：上一版最优基对新价格天然原始可行（解不变），
直接拿旧基重建规范型进第二阶段，价格小幅变化时通常零转轴即完成。
以下情况判定“基失效”，自动回退两阶段冷启动：
  1) 配方规格版本变化（约束集合/原料集合/营养集合可能改变）；
  2) 原料库结构变化导致行键集合（structure_key）不一致；
  3) 旧基重建规范型失败（基矩阵奇异）或重建后右端项出现负值。
structure_key 由配方版本 + 配方引用原料集合 + 行键序列构成，
价格变化不影响 key，因此跨价格版本可以复用基。
"""

from __future__ import annotations

import hashlib
import json

from ..solver.certificate import verify
from ..solver.diagnosis import diagnose
from ..solver.lp_builder import (
    Ingredient,
    IngredientLimit,
    NutrientBound,
    RatioConstraint,
    RecipeSpec,
    build_lp,
)
from ..solver.sensitivity import price_ranges
from ..solver.simplex import Simplex
from .storage import Storage


def structure_key(spec: RecipeSpec) -> str:
    """行结构指纹：与配方规格的约束布局绑定，价格不参与。"""

    payload = {
        "ingredients": [il.code for il in spec.ingredients],
        "bounds": [
            (il.code, il.min_ratio, il.max_ratio) for il in spec.ingredients
        ],
        "nutrients": [(n.code, n.lower, n.upper) for n in spec.nutrients],
        "ratios": [
            (r.numerator, r.denominator, r.lower, r.upper) for r in spec.ratios
        ],
        "total_mass": spec.total_mass,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]


def _to_domain_spec(spec_json: dict) -> RecipeSpec:
    return RecipeSpec(
        code=spec_json["code"],
        name=spec_json["name"],
        ingredients=tuple(
            IngredientLimit(
                code=i["code"],
                min_ratio=float(i["min_ratio"]),
                max_ratio=float(i["max_ratio"]),
            )
            for i in spec_json["ingredients"]
        ),
        nutrients=tuple(
            NutrientBound(code=n["code"], lower=n.get("lower"), upper=n.get("upper"))
            for n in spec_json.get("nutrients", [])
        ),
        ratios=tuple(
            RatioConstraint(
                numerator=r["numerator"],
                denominator=r["denominator"],
                lower=r.get("lower"),
                upper=r.get("upper"),
            )
            for r in spec_json.get("ratios", [])
        ),
        total_mass=float(spec_json.get("total_mass", 1.0)),
    )


class OptimizationService:
    def __init__(self, storage: Storage):
        self.db = storage

    # ------------------------------------------------------------- loading
    def _load(self, recipe_code: str, recipe_version: int | None, lib_version: int | None):
        rv = self.db.get_recipe_version(recipe_code, recipe_version)
        lib = self.db.get_library(lib_version)
        spec = _to_domain_spec(rv["spec"])
        catalog = {
            code: Ingredient(
                code=v["code"],
                name=v["name"],
                price=float(v["price"]),
                nutrients={k: float(x) for k, x in v["nutrients"].items()},
            )
            for code, v in lib["ingredients"].items()
        }
        known = set(self.db.nutrient_codes())
        return rv, lib, spec, catalog, known

    # ------------------------------------------------------------- optimize
    def optimize(
        self,
        recipe_code: str,
        recipe_version: int | None = None,
        lib_version: int | None = None,
        *,
        use_warm: bool = True,
        mode: str = "instant",
        job_id: int | None = None,
        persist: bool = True,
        save_basis: bool = True,
    ) -> dict:
        rv, lib, spec, catalog, known = self._load(
            recipe_code, recipe_version, lib_version
        )
        built = build_lp(spec, catalog, known)
        skey = structure_key(spec)

        warm_basis = None
        warm_source = None
        if use_warm:
            hit = self.db.get_warm_basis(recipe_code, skey)
            if hit is not None:
                warm_basis, warm_source = hit

        solver = Simplex(built.costs, built.rows)
        res = solver.solve(warm_basis=warm_basis)
        started_warm = warm_basis is not None
        warm_used = started_warm and res.status == "optimal"

        if res.status == "infeasible":
            diag = diagnose(built)
            report = {
                "status": "infeasible",
                "recipe_code": recipe_code,
                "recipe_version": rv["version"],
                "library_version": lib["version"],
                "diagnosis": diag,
                "warm_start": {"attempted": started_warm, "used": False},
            }
            result_id = None
            if persist:
                result_id = self.db.insert_result(
                    recipe_code=recipe_code,
                    recipe_version_id=rv["recipe_version_id"],
                    library_version_id=lib["id"],
                    status="infeasible",
                    cost=None,
                    report=report,
                    basis=None,
                    structure_key=skey,
                    mode=mode,
                    job_id=job_id,
                )
            report["result_id"] = result_id
            return report

        certificate = verify(built, res)
        ranges = price_ranges(built, res)
        amounts = {
            code: float(res.x[j]) for j, code in enumerate(built.ingredients)
        }
        achieved = {}
        assert built.nutrient_matrix is not None
        for r_, ncode in enumerate(built.nutrient_codes):
            achieved[ncode] = float(
                sum(built.nutrient_matrix[r_, j] * res.x[j] for j in range(len(res.x)))
            )

        report = {
            "status": "optimal",
            "recipe_code": recipe_code,
            "recipe_version": rv["version"],
            "recipe_version_id": rv["recipe_version_id"],
            "library_version": lib["version"],
            "library_version_id": lib["id"],
            "cost": res.objective,
            "amounts_kg": amounts,
            "achieved_nutrients": achieved,
            "constraints": certificate["rows"],
            "reduced_costs": certificate["columns"],
            "certificate": {
                k: certificate[k]
                for k in (
                    "primal_objective",
                    "dual_objective",
                    "absolute_gap",
                    "relative_gap",
                    "primal_feasible",
                    "dual_feasible",
                    "complementary_slackness",
                    "optimal_certificate_ok",
                )
            },
            "price_sensitivity": ranges,
            "binding_constraints": [
                r["key"] for r in certificate["rows"] if r["binding"]
            ],
            "shadow_prices": {
                r["key"]: r["dual"] for r in certificate["rows"]
            },
            "warm_start": {
                "attempted": started_warm,
                "used": warm_used,
                "source_library_version": warm_source,
                "fell_back_to_cold": started_warm and not warm_used,
            },
            "structure_key": skey,
        }

        result_id = None
        if persist:
            result_id = self.db.insert_result(
                recipe_code=recipe_code,
                recipe_version_id=rv["recipe_version_id"],
                library_version_id=lib["id"],
                status="optimal",
                cost=res.objective,
                report=report,
                basis=res.basis if save_basis else None,
                structure_key=skey,
                mode=mode,
                job_id=job_id,
                save_warm=save_basis,
            )
            if not save_basis:
                # 批处理进行中：热启动基随报告带回，作业成功结束时统一落库
                report["_basis"] = list(res.basis or [])
        report["result_id"] = result_id
        return report

    # ------------------------------------------------------------ compare
    def compare(self, result_a_id: int, result_b_id: int) -> dict:
        a = self.db.get_result(result_a_id)
        b = self.db.get_result(result_b_id)
        ra, rb = a["report"], b["report"]
        if a["recipe_code"] != b["recipe_code"]:
            raise ValueError("只能对比同一配方的两个结果")

        def blank(r: dict) -> dict:
            return {
                "status": r["status"],
                "library_version": r.get("library_version"),
                "recipe_version": r.get("recipe_version"),
                "cost": r.get("cost"),
                "amounts_kg": {},
            }

        if ra["status"] != "optimal" or rb["status"] != "optimal":
            return {
                "recipe_code": a["recipe_code"],
                "a": blank(ra),
                "b": blank(rb),
                "cost_delta": None,
                "amount_delta": {},
                "note": "其中至少一个结果不可行，仅对比状态",
            }
        codes = sorted(set(ra["amounts_kg"]) | set(rb["amounts_kg"]))
        delta = {
            c: round(rb["amounts_kg"].get(c, 0.0) - ra["amounts_kg"].get(c, 0.0), 12)
            for c in codes
        }
        return {
            "recipe_code": a["recipe_code"],
            "a": {
                "result_id": result_a_id,
                "library_version": ra["library_version"],
                "recipe_version": ra["recipe_version"],
                "cost": ra["cost"],
                "amounts_kg": ra["amounts_kg"],
            },
            "b": {
                "result_id": result_b_id,
                "library_version": rb["library_version"],
                "recipe_version": rb["recipe_version"],
                "cost": rb["cost"],
                "amounts_kg": rb["amounts_kg"],
            },
            "cost_delta": rb["cost"] - ra["cost"],
            "cost_delta_relative": (
                (rb["cost"] - ra["cost"]) / ra["cost"] if ra["cost"] else None
            ),
            "amount_delta": delta,
        }
