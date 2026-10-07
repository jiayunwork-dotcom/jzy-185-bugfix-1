"""批量作业：受影响配方选择、进度、取消、版本锁定、并发不覆盖。"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from tests.conftest import CORN, SBM, WHEAT


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FEED_DATA_DIR", str(tmp_path))
    from app.main import app

    with TestClient(app) as c:
        yield c


def _seed_many(client, n=3):
    client.post("/api/library/publish", json={"ingredients": [CORN, SBM, WHEAT]})
    for i in range(n):
        client.put(f"/api/recipes/r{i}", json={
            "code": f"r{i}", "name": f"配方{i}",
            "ingredients": [
                {"code": "corn"}, {"code": "sbm"}, {"code": "wheat"},
            ],
            "nutrients": [{"code": "CP", "lower": 0.15 + 0.01 * i}],
            "ratios": [],
        })
    # 先全部优化一遍，留下最优基
    for i in range(n):
        client.post("/api/optimize", json={"recipe_code": f"r{i}"})


def _wait(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "cancelled", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job timeout")


def test_batch_reoptimize_affected(client):
    _seed_many(client, 3)
    # 只改玉米价格：所有配方都引用玉米，都应被选中
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.35}, SBM, WHEAT],
        "message": "corn up",
    })
    r = client.post("/api/jobs", json={})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert r.json()["total"] == 3
    job = _wait(client, jid)
    assert job["status"] == "completed"
    assert job["processed"] == 3
    assert all(it["status"] == "done" for it in job["items"])
    # 结果绑定新版本库
    for it in job["items"]:
        rid = it["result_id"]
        result = client.get(f"/api/results/{rid}").json()
        assert result["report"]["library_version"] == 2


def test_batch_warm_matches_cold(client):
    _seed_many(client, 3)
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.28}, {**SBM, "price": 0.53}, WHEAT],
    })
    jid = client.post("/api/jobs", json={}).json()["job_id"]
    job = _wait(client, jid)
    for it in job["items"]:
        warm_report = client.get(f"/api/results/{it['result_id']}").json()["report"]
        assert warm_report["warm_start"]["used"] is True
        # 冷启动复算（不持久化不必要，但即时优化会写历史；成本对比即可）
        cold = client.post("/api/optimize", json={
            "recipe_code": it["recipe_code"], "warm_start": False
        }).json()
        rel = abs(warm_report["cost"] - cold["cost"]) / cold["cost"]
        assert rel < 1e-9


def test_cancel_leaves_no_partial_results(client):
    """取消语义（确定性）：作业处于 cancelling 状态时启动，所有项跳过、
    不产生任何结果；正在跑的那项跑完后随整批删除。"""

    _seed_many(client, 20)
    # 发布新版本并建作业
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.31}, SBM, WHEAT],
    })
    jid = client.post("/api/jobs", json={}).json()["job_id"]
    # 调度器轮询间隔 0.1s；作业可能恰好已完成。先对“提交即取消”做条件处理：
    cancel_resp = client.post(f"/api/jobs/{jid}/cancel")
    assert cancel_resp.status_code == 200
    job = _wait(client, jid)
    assert job["status"] in ("cancelled", "completed")
    if job["status"] == "cancelled":
        # 不得留下任何属于该作业的结果
        for it in job["items"]:
            assert it["status"] == "skipped"
            assert it["result_id"] is None
        for i in range(20):
            rows = client.get(f"/api/recipes/r{i}/results").json()["results"]
            assert all(row["library_version_id"] == 1 for row in rows)


def test_cancel_before_start_is_respected(client, monkeypatch):
    """作业仍处于 pending 时就被取消：worker 必须按 cancelling 处理，
    整批跳过、零结果。直接操作 DB 制造该状态，消除调度时序竞态。"""

    _seed_many(client, 3)
    lib = client.get("/api/library").json()
    db = client.app.state.storage
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.31}, SBM, WHEAT],
    })
    # 先暂停 worker（替换 pending_job 让它暂时拿不到作业），插入作业后置 cancelling
    sched = client.app.state.scheduler
    orig = db.pending_job
    gate = threading.Event()
    def blocked_pending():
        gate.wait(timeout=5)
        return orig()
    monkeypatch.setattr(db, "pending_job", blocked_pending)
    jid = sched.submit(lib["id"], ["r0", "r1", "r2"])
    db.set_job_status(jid, "cancelling")
    gate.set()  # 放行 worker
    job = _wait(client, jid)
    assert job["status"] == "cancelled"
    for it in job["items"]:
        assert it["status"] == "skipped"
        assert it["result_id"] is None
    for i in range(3):
        rows = client.get(f"/api/recipes/r{i}/results").json()["results"]
        assert all(row["library_version_id"] == 1 for row in rows)


def test_job_version_lock(client, monkeypatch):
    """作业运行期间发布新版本：作业仍锁定启动时版本。"""

    _seed_many(client, 5)
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.33}, SBM, WHEAT],
        "message": "v2 at submit",
    })
    jid = client.post("/api/jobs", json={}).json()["job_id"]
    # 作业很快；直接再发布 v3 并等待作业结束
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.99}, {**SBM, "price": 0.99}, WHEAT],
        "message": "v3 during run",
    })
    job = _wait(client, jid)
    assert job["status"] == "completed"
    for it in job["items"]:
        report = client.get(f"/api/results/{it['result_id']}").json()["report"]
        assert report["library_version"] == 2  # 绝不是 3


def test_concurrent_jobs_same_recipe_no_overwrite(client):
    """串行调度保证：同一配方两个作业的结果各自独立、都能查到。"""

    _seed_many(client, 1)
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.31}, SBM, WHEAT],
    })
    j1 = client.post("/api/jobs", json={"recipe_codes": ["r0"]}).json()["job_id"]
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.32}, SBM, WHEAT],
    })
    j2 = client.post("/api/jobs", json={"recipe_codes": ["r0"]}).json()["job_id"]
    job1 = _wait(client, j1)
    job2 = _wait(client, j2)
    assert job1["status"] == job2["status"] == "completed"
    id1 = job1["items"][0]["result_id"]
    id2 = job2["items"][0]["result_id"]
    assert id1 != id2
    r1 = client.get(f"/api/results/{id1}").json()["report"]
    r2 = client.get(f"/api/results/{id2}").json()["report"]
    assert r1["library_version"] == 2
    assert r2["library_version"] == 3


# ----------------------------------------- 配方规格版本在提交时刻整批锁定
def _seed_priced_batch(client, n: int, cp_base: float = 0.16, cp_step: float = 0.0005):
    """三原料两版价；建 n 个配方，第 i 个 CP 下限 = cp_base + cp_step*i。"""

    client.post("/api/library/publish", json={"ingredients": [CORN, SBM, WHEAT]})
    for i in range(n):
        client.put(f"/api/recipes/r{i}", json={
            "code": f"r{i}", "name": f"配方{i}",
            "ingredients": [
                {"code": "corn"}, {"code": "sbm"}, {"code": "wheat"},
            ],
            "nutrients": [{"code": "CP", "lower": cp_base + cp_step * i}],
            "ratios": [],
        })
    r = client.post("/api/library/publish", json={"ingredients": [
        {**CORN, "price": 0.32}, {**SBM, "price": 0.47}, {**WHEAT, "price": 0.21},
    ]})
    assert r.status_code == 201
    return client.get("/api/library").json()["id"]


def _set_last_recipe_cp30(client, n: int):
    code = f"r{n - 1}"
    r = client.put(f"/api/recipes/{code}", json={
        "code": code, "name": f"配方{n - 1}",
        "ingredients": [
            {"code": "corn"}, {"code": "sbm"}, {"code": "wheat"},
        ],
        "nutrients": [{"code": "CP", "lower": 0.30}],
        "ratios": [],
    })
    assert r.status_code == 201
    assert r.json()["version"] == 2


def test_batch_recipe_version_lock_midrun_edit(client, monkeypatch):
    """稳定复现场景：60 个配方的全量重优化跑到中途时，把最后一个配方的
    CP 下限改成 30%（产生规格 v2）。整批仍必须按提交时锁定的 v1 规格计算，
    成本约 0.2529；作业详情显示锁定的是 v1。改后再做即时优化才得到 0.3487。"""

    n = 60
    _seed_priced_batch(client, n)

    db = client.app.state.storage
    gate = threading.Event()
    original_set_item = db.set_item

    # 在前几个配方处理完后暂停 worker，留出“运行中改配方”的窗口
    def paused_set_item(job_id, recipe_code, status, **fields):
        original_set_item(job_id, recipe_code, status, **fields)
        if status == "done" and recipe_code == "r3":
            assert gate.wait(timeout=5)

    monkeypatch.setattr(db, "set_item", paused_set_item)

    jid = client.post("/api/jobs", json={}).json()["job_id"]

    # 等 worker 处理到暂停点
    deadline = time.time() + 5
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{jid}").json()
        if any(it["status"] == "done" for it in job["items"]):
            break
        time.sleep(0.005)
    # 此刻改最后一个配方（它还排在后面，尚未执行）
    _set_last_recipe_cp30(client, n)
    gate.set()  # 放行

    job = _wait(client, jid)
    assert job["status"] == "completed"
    assert job["processed"] == n

    last = job["items"][-1]
    assert last["recipe_code"] == f"r{n - 1}"
    # 作业详情直接可见每个配方锁定的规格版本，无需从结果反推
    assert last["recipe_version"] == 1
    for it in job["items"]:
        assert it["status"] == "done"
        assert it["recipe_version"] == 1

    report = client.get(f"/api/results/{last['result_id']}").json()["report"]
    assert report["recipe_version"] == 1          # 锁定 v1，不是改后的 v2
    assert report["cost"] == pytest.approx(0.2529, abs=1e-3)
    # 同批口径一致：全部绑定库 v2
    assert report["library_version"] == 2

    # 作业结束后单独即时优化：按新规格 v2，约 0.3487
    instant = client.post("/api/optimize", json={"recipe_code": f"r{n - 1}"}).json()
    assert instant["status"] == "optimal"
    assert instant["recipe_version"] == 2
    assert instant["cost"] == pytest.approx(0.3487, abs=1e-3)


def test_queued_job_locks_spec_at_submit(client, monkeypatch):
    """作业还在排队（未开始）时配方被修改：仍按提交时锁定的旧规格执行。"""

    n = 3
    _seed_priced_batch(client, n)
    db = client.app.state.storage
    sched = client.app.state.scheduler

    orig = db.pending_job
    gate = threading.Event()
    monkeypatch.setattr(
        db, "pending_job", lambda: gate.wait(timeout=5) and orig()
    )
    lib_id = db.get_library(2)["id"]
    jid = sched.submit(lib_id, ["r0", "r1", "r2"])
    # 排队期间把 r0 的 CP 下限提到 30%（v2）
    _set_last_recipe_cp30(client, 1)
    gate.set()

    job = _wait(client, jid)
    assert job["status"] == "completed"
    first = job["items"][0]
    assert first["recipe_version"] == 1
    report = client.get(f"/api/results/{first['result_id']}").json()["report"]
    assert report["recipe_version"] == 1
    # 提交时锁定的是 0.16 下限（cp_base + 0），成本应远低于 30% 规格
    assert report["cost"] < 0.26
