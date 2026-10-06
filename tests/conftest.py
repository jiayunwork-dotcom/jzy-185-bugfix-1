"""pytest 公共夹具：每个测试独立临时数据库。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.optimizer import OptimizationService
from app.services.storage import Storage


CORN = {"code": "corn", "name": "玉米", "price": 0.30,
        "nutrients": {"DM": 0.88, "CP": 0.08, "ME": 3.35, "CA": 0.001,
                      "TP": 0.003, "LYS": 0.0026, "MET": 0.0019}}
SBM = {"code": "sbm", "name": "豆粕", "price": 0.50,
       "nutrients": {"DM": 0.89, "CP": 0.44, "ME": 2.55, "CA": 0.003,
                     "TP": 0.006, "LYS": 0.027, "MET": 0.0065}}
WHEAT = {"code": "wheat", "name": "麸皮", "price": 0.20,
         "nutrients": {"DM": 0.87, "CP": 0.14, "ME": 2.0, "CA": 0.001,
                       "TP": 0.009, "LYS": 0.006, "MET": 0.0023}}


@pytest.fixture
def db(tmp_path):
    return Storage(str(tmp_path / "test.db"))


@pytest.fixture
def opt(db):
    return OptimizationService(db)


@pytest.fixture
def lib_v1(db):
    return db.publish_library([CORN, SBM, WHEAT], "v1")


def recipe_spec(
    ingredients=("corn", "sbm"),
    bounds=None,
    nutrients=(("CP", 0.18, None),),
    ratios=(),
    total=1.0,
):
    """构造配方规格 dict。bounds: {code:(lo,hi)}。"""

    b = bounds or {}
    return {
        "code": "r1",
        "name": "测试配方",
        "ingredients": [
            {"code": c, "min_ratio": b.get(c, (0.0, 1.0))[0],
             "max_ratio": b.get(c, (0.0, 1.0))[1]}
            for c in ingredients
        ],
        "nutrients": [
            {"code": n, "lower": lo, "upper": hi} for n, lo, hi in nutrients
        ],
        "ratios": [
            {"numerator": a, "denominator": d, "lower": lo, "upper": hi}
            for a, d, lo, hi in ratios
        ],
        "total_mass": total,
    }
