"""在线升级：旧数据卷（历史作业 + 排队作业）补齐配方版本锁列并回填。"""

from __future__ import annotations

import sqlite3

from tests.conftest import CORN, SBM, WHEAT

from app.services.storage import Storage


def _old_schema_sql() -> str:
    """升级前 job_items 不含 recipe_version_id / recipe_version 两列。"""

    return """
    CREATE TABLE job_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL REFERENCES jobs(id),
        recipe_code TEXT NOT NULL,
        seq INTEGER NOT NULL,
        status TEXT NOT NULL,
        result_id INTEGER,
        error TEXT NOT NULL DEFAULT '',
        UNIQUE (job_id, recipe_code)
    );
    """


def test_migration_backfills_historical_and_queued_jobs(tmp_path):
    path = str(tmp_path / "feedmill.db")
    db = Storage(path)

    db.publish_library([CORN, SBM, WHEAT], "v1")
    db.create_recipe("a", "A", {"code": "a", "name": "A", "ingredients": [
        {"code": "corn", "min_ratio": 0.0, "max_ratio": 1.0},
        {"code": "sbm", "min_ratio": 0.0, "max_ratio": 1.0},
        {"code": "wheat", "min_ratio": 0.0, "max_ratio": 1.0},
    ], "nutrients": [{"code": "CP", "lower": 0.16}], "ratios": []})
    db.create_recipe("b", "B", {"code": "b", "name": "B", "ingredients": [
        {"code": "corn", "min_ratio": 0.0, "max_ratio": 1.0},
        {"code": "sbm", "min_ratio": 0.0, "max_ratio": 1.0},
        {"code": "wheat", "min_ratio": 0.0, "max_ratio": 1.0},
    ], "nutrients": [{"code": "CP", "lower": 0.17}], "ratios": []})

    # 历史完成作业：对 a 做过即时优化（绑定 a 的 v1），并造一条作业结果
    lib1 = db.get_library(1)
    from app.services.optimizer import OptimizationService

    opt = OptimizationService(db)
    a_v1_report = opt.optimize("a", mode="batch", job_id=None)
    a_v1_rvid = a_v1_report["recipe_version_id"]

    done_jid = db.create_job(lib1["id"], ["a"])  # 新版 Storage 已带锁列
    db.set_job_status(done_jid, "completed", finished_at=1.0)
    db.set_item(done_jid, "a", "done", result_id=a_v1_report["result_id"])

    # a 之后又改过一版规格
    db.create_recipe("a", "A", {"code": "a", "name": "A", "ingredients": [
        {"code": "corn", "min_ratio": 0.0, "max_ratio": 1.0},
        {"code": "sbm", "min_ratio": 0.0, "max_ratio": 1.0},
        {"code": "wheat", "min_ratio": 0.0, "max_ratio": 1.0},
    ], "nutrients": [{"code": "CP", "lower": 0.30}], "ratios": []}, "CP 30")

    # 排队作业（含一个将被跳过/失败的无结果项 c 不存在于任何规格——改用 b）
    pending_jid = db.create_job(lib1["id"], ["b"])
    db.set_item(pending_jid, "b", "skipped")  # 模拟无结果项

    # 把库降级为“旧卷”：去掉锁列并按旧表结构重建（保留 jobs 与结果数据）
    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA foreign_keys=OFF")
        raw.execute("ALTER TABLE job_items RENAME TO job_items_new")
        raw.executescript(_old_schema_sql())
        raw.execute(
            "INSERT INTO job_items(id,job_id,recipe_code,seq,status,result_id,error)"
            " SELECT id,job_id,recipe_code,seq,status,result_id,error"
            " FROM job_items_new"
        )
        raw.execute("DROP TABLE job_items_new")

    # 重新打开：触发在线迁移
    del db
    db2 = Storage(path)

    # 历史完成项：回填为结果实际绑定的版本（a 的 v1，而不是当前最新 v2）
    done_job = db2.get_job(done_jid)
    done_item = done_job["items"][0]
    assert done_item["recipe_version_id"] == a_v1_rvid
    assert done_item["recipe_version"] == 1

    # 无结果项：回填为当时最新版本（b 仍是 v1）
    pending_job = db2.get_job(pending_jid)
    pending_item = pending_job["items"][0]
    assert pending_item["recipe_version"] == 1
    assert pending_item["recipe_version_id"] is not None

    # 排队作业仍能被 worker 正常捞取（按锁定版本执行/收尾），不会卡死
    assert db2.pending_job() == pending_jid
