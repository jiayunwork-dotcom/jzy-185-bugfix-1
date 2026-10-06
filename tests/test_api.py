"""HTTP 接口测试。"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FEED_DATA_DIR", str(tmp_path))
    # 延迟导入，确保环境变量生效（db 路径在 Storage 实例化时读取）
    from app.main import app

    with TestClient(app) as c:
        yield c


CORN = {"code": "corn", "name": "玉米", "price": 0.30,
        "nutrients": {"DM": 0.88, "CP": 0.08, "CA": 0.001, "TP": 0.003}}
SBM = {"code": "sbm", "name": "豆粕", "price": 0.50,
       "nutrients": {"DM": 0.89, "CP": 0.44, "CA": 0.003, "TP": 0.006}}

RECIPE = {
    "code": "r1", "name": "蛋鸡料",
    "ingredients": [
        {"code": "corn", "min_ratio": 0, "max_ratio": 1},
        {"code": "sbm", "min_ratio": 0, "max_ratio": 1},
    ],
    "nutrients": [{"code": "CP", "lower": 0.18}],
    "ratios": [], "total_mass": 1.0,
}


def seed(client):
    r = client.post("/api/library/publish",
                    json={"ingredients": [CORN, SBM], "message": "v1"})
    assert r.status_code == 201, r.text
    r = client.put("/api/recipes/r1", json=RECIPE)
    assert r.status_code == 201, r.text


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_validation_errors_carry_field_names(client):
    r = client.post("/api/library/publish", json={"ingredients": [
        {"code": "bad", "name": "坏", "price": -3,
         "nutrients": {"DM": 0.9, "X": 0.1}},
    ]})
    assert r.status_code == 422
    fields = {e["field"] for e in r.json()["detail"]}
    assert "ingredients[0].price" in fields
    assert "ingredients[0].nutrients.X" in fields


def test_optimize_endpoint_and_history(client):
    seed(client)
    r = client.post("/api/optimize", json={"recipe_code": "r1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "optimal"
    assert body["amounts_kg"]["sbm"] == pytest.approx(0.27777778, abs=1e-6)
    assert body["cost"] == pytest.approx(0.3555555556, abs=1e-9)
    assert body["certificate"]["optimal_certificate_ok"]

    hist = client.get("/api/recipes/r1/results").json()["results"]
    assert len(hist) == 1

    r2 = client.get(f"/api/results/{body['result_id']}")
    assert r2.status_code == 200
    assert r2.json()["report"]["library_version"] == 1


def test_recipe_versioning_and_compare(client):
    seed(client)
    a = client.post("/api/optimize", json={"recipe_code": "r1"}).json()
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.32}, {**SBM, "price": 0.52}]
    })
    b = client.post("/api/optimize", json={"recipe_code": "r1"}).json()
    cmp_ = client.post("/api/compare", json={
        "recipe_code": "r1", "result_a": a["result_id"], "result_b": b["result_id"]
    }).json()
    assert cmp_["cost_delta"] == pytest.approx(0.02, abs=1e-9)
    versions = client.get("/api/recipes/r1/versions").json()["versions"]
    assert len(versions) == 1  # 价格更新不改配方版本


def test_infeasible_api_conflict(client):
    seed(client)
    r = client.put("/api/recipes/r2", json={
        "code": "r2", "name": "不可行",
        "ingredients": [
            {"code": "corn", "min_ratio": 0, "max_ratio": 0.4},
            {"code": "sbm", "min_ratio": 0, "max_ratio": 0.4},
        ],
        "nutrients": [], "ratios": [],
    })
    assert r.status_code == 201
    body = client.post("/api/optimize", json={"recipe_code": "r2"}).json()
    assert body["status"] == "infeasible"
    assert body["diagnosis"]["reason_code"] == "max_sum_below_total"
    assert len(body["diagnosis"]["conflicts"]) >= 2
