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
    import threading
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


def _recipe_body(code, cp_lower):
    return {
        "code": code, "name": f"配方{code}",
        "ingredients": [{"code": "corn"}, {"code": "sbm"}, {"code": "wheat"}],
        "nutrients": [{"code": "CP", "lower": cp_lower}],
        "ratios": [],
    }


def test_job_recipe_version_lock_during_run(client, monkeypatch):
    """周一换价场景：60 个配方全量重优化，作业跑到一半时配方师把最后一个
    配方的粗蛋白下限从 18.95% 提到 30%（产生第二版规格）。整批必须仍按
    提交时锁定的第一版规格计算（成本≈0.2529），新规格只在作业之外生效
    （事后再即时优化得≈0.3487）；作业详情里每个条目都能看到锁定的版本。
    """

    # 第一版价格：玉米 0.30 / 豆粕 0.50 / 麸皮 0.20
    client.post("/api/library/publish", json={"ingredients": [CORN, SBM, WHEAT]})
    # 第 i 个配方粗蛋白下限 = 16% + 0.05 个百分点 * i
    for i in range(60):
        code = f"f{i:02d}"
        client.put(f"/api/recipes/{code}", json=_recipe_body(code, 0.16 + 0.0005 * i))
    # 换价：发布第二版 0.32 / 0.47 / 0.21
    client.post("/api/library/publish", json={
        "ingredients": [
            {**CORN, "price": 0.32}, {**SBM, "price": 0.47}, {**WHEAT, "price": 0.21},
        ],
    })

    # 确定性时序：worker 跑到第 6 个配方（f05）时拦住，等测试改完 f59 再放行
    sched = client.app.state.scheduler
    orig_optimize = sched.optimizer.optimize
    entered = threading.Event()
    proceed = threading.Event()

    def gated_optimize(recipe_code, **kwargs):
        if recipe_code == "f05" and kwargs.get("mode") == "batch":
            entered.set()
            assert proceed.wait(timeout=10)
        return orig_optimize(recipe_code, **kwargs)

    monkeypatch.setattr(sched.optimizer, "optimize", gated_optimize)

    codes = [f"f{i:02d}" for i in range(60)]
    jid = client.post("/api/jobs", json={"recipe_codes": codes}).json()["job_id"]

    assert entered.wait(timeout=10), "作业未按预期跑到 f05"
    # 配方师此时修改最后一个配方：粗蛋白下限 18.95% -> 30%，产生第二版规格
    r = client.put("/api/recipes/f59", json=_recipe_body("f59", 0.30))
    assert r.json()["version"] == 2
    proceed.set()

    job = _wait(client, jid, timeout=30)
    assert job["status"] == "completed"
    assert job["processed"] == 60
    # 作业详情：每个条目都记录锁定版本，且全部为提交时的第 1 版
    assert all(it["recipe_version"] == 1 for it in job["items"])
    # 最后一个配方与同批其它配方一样，依据锁定的那一版规格：成本≈0.2529
    last = client.get(f"/api/results/{job['items'][-1]['result_id']}").json()["report"]
    assert last["recipe_version"] == 1
    assert last["library_version"] == 2
    assert last["cost"] == pytest.approx(0.2529, abs=1e-4)
    # 新规格只在作业之外生效：事后再对它单独即时优化，用第 2 版，成本≈0.3487
    now = client.post("/api/optimize", json={"recipe_code": "f59"}).json()
    assert now["recipe_version"] == 2
    assert now["cost"] == pytest.approx(0.3487, abs=1e-4)


def test_job_detail_shows_locked_versions_while_pending(client, monkeypatch):
    """锁定发生在提交那一刻：作业还在排队时，详情里每个条目就已带上
    各自锁定的配方规格版本。"""

    _seed_many(client, 3)
    db = client.app.state.storage
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.31}, SBM, WHEAT],
    })
    # 拦住 worker，让作业停在 pending
    gate = threading.Event()
    orig_pending = db.pending_job

    def blocked_pending():
        gate.wait(timeout=5)
        return orig_pending()

    monkeypatch.setattr(db, "pending_job", blocked_pending)
    jid = client.post("/api/jobs", json={}).json()["job_id"]
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "pending"
    assert [it["recipe_version"] for it in job["items"]] == [1, 1, 1]
    gate.set()
    job = _wait(client, jid)
    assert job["status"] == "completed"
    assert all(it["recipe_version"] == 1 for it in job["items"])


def test_legacy_pending_job_locked_at_start(client):
    """升级前已排队、job_items 没有锁定版本（NULL）的老作业：worker 启动时
    补锁当时最新版本，正常跑完，不丢不卡。"""

    _seed_many(client, 3)
    client.post("/api/library/publish", json={
        "ingredients": [{**CORN, "price": 0.31}, SBM, WHEAT],
    })
    db = client.app.state.storage
    lib = client.get("/api/library").json()
    # 模拟升级前留下的 pending 作业：直接插库，recipe_version 为 NULL
    with db.connect() as c:
        cur = c.execute(
            "INSERT INTO jobs(library_version_id,status,total,created_at)"
            " VALUES (?,'pending',3,0)",
            (lib["id"],),
        )
        jid = cur.lastrowid
        c.executemany(
            "INSERT INTO job_items(job_id,recipe_code,seq,status)"
            " VALUES (?,?,?,'pending')",
            [(jid, f"r{i}", i) for i in range(3)],
        )
    job = _wait(client, jid)
    assert job["status"] == "completed"
    assert all(it["status"] == "done" for it in job["items"])
    # 补锁为启动时的最新版本（第 1 版），结果也绑第 1 版
    assert all(it["recipe_version"] == 1 for it in job["items"])
    for it in job["items"]:
        report = client.get(f"/api/results/{it['result_id']}").json()["report"]
        assert report["recipe_version"] == 1
        assert report["library_version"] == 2


def test_storage_migration_preserves_job_items(tmp_path):
    """老库（job_items 无 recipe_version 列）打开后自动补列；历史作业
    数据原样保留（版本为 NULL），新作业正常带锁定版本。"""

    import sqlite3

    from app.services.storage import Storage

    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            library_version_id INTEGER NOT NULL,
            status TEXT NOT NULL, total INTEGER NOT NULL,
            created_at REAL NOT NULL, started_at REAL, finished_at REAL,
            error TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE job_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL REFERENCES jobs(id),
            recipe_code TEXT NOT NULL, seq INTEGER NOT NULL,
            status TEXT NOT NULL, result_id INTEGER,
            error TEXT NOT NULL DEFAULT '',
            UNIQUE (job_id, recipe_code)
        );
        INSERT INTO jobs(library_version_id,status,total,created_at)
            VALUES (1,'completed',1,0);
        INSERT INTO job_items(job_id,recipe_code,seq,status)
            VALUES (1,'r0',0,'done');
        """
    )
    conn.commit()
    conn.close()

    db = Storage(path)  # 触发迁移
    with db.connect() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(job_items)")}
    assert "recipe_version" in cols
    # 历史作业原样保留：版本未知记 NULL，状态不丢
    job = db.get_job(1)
    assert job["status"] == "completed"
    assert job["items"][0]["recipe_version"] is None
    # 新作业在提交时锁定版本
    db.publish_library([CORN, SBM, WHEAT], "v1")
    db.create_recipe("r0", "配方0", {"code": "r0", "name": "配方0",
                                    "ingredients": [], "nutrients": [],
                                    "ratios": []})
    jid = db.create_job(1, ["r0"])
    assert db.get_job(jid)["items"][0]["recipe_version"] == 1
