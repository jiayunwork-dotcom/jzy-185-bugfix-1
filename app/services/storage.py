"""SQLite 存储层：原料库/配方的版本化、优化结果与作业持久化。

所有版本内容均为不可变快照（append-only）。结果表只追加不更新，
并发作业不会互相覆盖；同一配方同一版本的结果以 (recipe_version_id,
library_version_id, mode) 唯一键去重。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any

from ..config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS nutrients (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    unit TEXT NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL DEFAULT 0,
    builtin INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS library_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    ingredient_codes TEXT NOT NULL          -- JSON 快照顺序
);

CREATE TABLE IF NOT EXISTS ingredient_versions (
    library_version_id INTEGER NOT NULL REFERENCES library_versions(id),
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    nutrients_json TEXT NOT NULL,
    PRIMARY KEY (library_version_id, code)
);

CREATE TABLE IF NOT EXISTS recipes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS recipe_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id INTEGER NOT NULL REFERENCES recipes(id),
    version INTEGER NOT NULL,
    created_at REAL NOT NULL,
    spec_json TEXT NOT NULL,                -- 完整规格快照
    message TEXT NOT NULL DEFAULT '',
    UNIQUE (recipe_id, version)
);

CREATE TABLE IF NOT EXISTS optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_code TEXT NOT NULL,
    recipe_version_id INTEGER NOT NULL,
    library_version_id INTEGER NOT NULL,
    status TEXT NOT NULL,                   -- optimal / infeasible
    cost REAL,
    report_json TEXT NOT NULL,
    basis_json TEXT,                        -- 热启动基（仅 optimal）
    structure_key TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'instant',   -- instant / batch
    job_id INTEGER,
    created_at REAL NOT NULL,
    UNIQUE (recipe_version_id, library_version_id, mode, job_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    library_version_id INTEGER NOT NULL,
    status TEXT NOT NULL,                   -- pending/running/cancelling/cancelled/completed/failed
    total INTEGER NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS job_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    recipe_code TEXT NOT NULL,
    seq INTEGER NOT NULL,
    status TEXT NOT NULL,                   -- pending/running/done/failed/skipped
    recipe_version INTEGER,                 -- 提交时锁定的配方规格版本（升级前数据为 NULL）
    result_id INTEGER,
    error TEXT NOT NULL DEFAULT '',
    UNIQUE (job_id, recipe_code)
);

CREATE TABLE IF NOT EXISTS warm_basis (
    recipe_code TEXT NOT NULL,
    structure_key TEXT NOT NULL,
    basis_json TEXT NOT NULL,
    library_version_id INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (recipe_code, structure_key)
);
"""

BUILTIN_NUTRIENTS = [
    ("DM", "干物质", "", 0, 1),
    ("CP", "粗蛋白", "", 1, 1),
    ("ME", "代谢能", "Mcal/kg", 2, 1),
    ("CA", "钙", "", 3, 1),
    ("TP", "总磷", "", 4, 1),
    ("LYS", "赖氨酸", "", 5, 1),
    ("MET", "蛋氨酸", "", 6, 1),
]


class Storage:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.resolved_db_path()
        self._lock = threading.RLock()
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)
            self._migrate(conn)
            conn.executemany(
                "INSERT OR IGNORE INTO nutrients(code,name,unit,sort_order,builtin)"
                " VALUES (?,?,?,?,?)",
                BUILTIN_NUTRIENTS,
            )

    @staticmethod
    def _migrate(conn):
        """老库就地升级（服务在线、数据卷内有历史与排队作业，只增不改）。"""

        cols = {r["name"] for r in conn.execute("PRAGMA table_info(job_items)")}
        if "recipe_version" not in cols:
            # 既有行保持 NULL：历史作业不再追溯；排队作业由 worker 启动时补锁
            conn.execute(
                "ALTER TABLE job_items ADD COLUMN recipe_version INTEGER"
            )

    @contextmanager
    def connect(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ----------------------------------------------------------- nutrients
    def list_nutrients(self) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT * FROM nutrients ORDER BY sort_order, code"
            ).fetchall()
            return [dict(r) for r in rows]

    def add_nutrient(self, code: str, name: str, unit: str = "") -> dict:
        with self.connect() as c:
            c.execute(
                "INSERT INTO nutrients(code,name,unit,sort_order,builtin) "
                "VALUES (?,?,?,0,0)",
                (code, name, unit),
            )
        return {"code": code, "name": name, "unit": unit}

    def nutrient_codes(self) -> set[str]:
        with self.connect() as c:
            return {r["code"] for r in c.execute("SELECT code FROM nutrients")}

    # ------------------------------------------------------------- library
    def latest_library_version(self) -> int | None:
        with self.connect() as c:
            r = c.execute("SELECT MAX(version) v FROM library_versions").fetchone()
            return r["v"]

    def publish_library(
        self, ingredients: list[dict], message: str = ""
    ) -> int:
        """ingredients: [{code,name,price,nutrients:{code:value}}]。全量快照发布。"""

        with self.connect() as c:
            v = (self.latest_library_version() or 0) + 1
            codes = [i["code"] for i in ingredients]
            c.execute(
                "INSERT INTO library_versions(version,created_at,message,ingredient_codes)"
                " VALUES (?,?,?,?)",
                (v, time.time(), message, json.dumps(codes, ensure_ascii=False)),
            )
            vid = c.execute(
                "SELECT id FROM library_versions WHERE version=?", (v,)
            ).fetchone()["id"]
            c.executemany(
                "INSERT INTO ingredient_versions(library_version_id,code,name,price,nutrients_json)"
                " VALUES (?,?,?,?,?)",
                [
                    (
                        vid,
                        i["code"],
                        i["name"],
                        float(i["price"]),
                        json.dumps(i["nutrients"], ensure_ascii=False),
                    )
                    for i in ingredients
                ],
            )
            return v

    def get_library(self, version: int | None = None) -> dict:
        with self.connect() as c:
            if version is None:
                r = c.execute(
                    "SELECT * FROM library_versions ORDER BY version DESC LIMIT 1"
                ).fetchone()
            else:
                r = c.execute(
                    "SELECT * FROM library_versions WHERE version=?", (version,)
                ).fetchone()
            if r is None:
                raise LookupError(f"原料库版本不存在: {version}")
            items = c.execute(
                "SELECT code,name,price,nutrients_json FROM ingredient_versions"
                " WHERE library_version_id=?",
                (r["id"],),
            ).fetchall()
            ingredients = {
                x["code"]: {
                    "code": x["code"],
                    "name": x["name"],
                    "price": x["price"],
                    "nutrients": json.loads(x["nutrients_json"]),
                }
                for x in items
            }
            return {
                "version": r["version"],
                "id": r["id"],
                "created_at": r["created_at"],
                "message": r["message"],
                "ingredient_codes": json.loads(r["ingredient_codes"]),
                "ingredients": ingredients,
            }

    def get_library_version_by_id(self, version_id: int) -> dict:
        with self.connect() as c:
            r = c.execute(
                "SELECT * FROM library_versions WHERE id=?", (version_id,)
            ).fetchone()
            if r is None:
                raise LookupError(f"原料库版本 id 不存在: {version_id}")
            return {"id": r["id"], "version": r["version"]}

    def list_library_versions(self) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT id,version,created_at,message FROM library_versions"
                " ORDER BY version"
            ).fetchall()
            return [dict(r) for r in rows]

    def diff_library_versions(self, old: int, new: int) -> dict[str, list[str]]:
        """两个版本间价格或营养值发生变化的原料代码（新增/删除也计入）。"""

        a = self.get_library(old)["ingredients"]
        b = self.get_library(new)["ingredients"]
        changed: list[str] = []
        for code in sorted(set(a) | set(b)):
            if code not in a or code not in b:
                changed.append(code)
            elif (
                abs(a[code]["price"] - b[code]["price"]) > 1e-12
                or a[code]["nutrients"] != b[code]["nutrients"]
            ):
                changed.append(code)
        return {"changed_ingredients": changed}

    # -------------------------------------------------------------- recipe
    def create_recipe(self, code: str, name: str, spec: dict, message: str = "") -> int:
        with self.connect() as c:
            now = time.time()
            c.execute(
                "INSERT OR IGNORE INTO recipes(code,name,created_at) VALUES (?,?,?)",
                (code, name, now),
            )
            rid = c.execute("SELECT id FROM recipes WHERE code=?", (code,)).fetchone()[
                "id"
            ]
            c.execute(
                "UPDATE recipes SET name=? WHERE id=?", (name, rid)
            )
            v = (
                c.execute(
                    "SELECT MAX(version) v FROM recipe_versions WHERE recipe_id=?",
                    (rid,),
                ).fetchone()["v"]
                or 0
            ) + 1
            c.execute(
                "INSERT INTO recipe_versions(recipe_id,version,created_at,spec_json,message)"
                " VALUES (?,?,?,?,?)",
                (rid, v, now, json.dumps(spec, ensure_ascii=False), message),
            )
            return v

    def get_recipe_version(
        self, code: str, version: int | None = None
    ) -> dict:
        with self.connect() as c:
            r = c.execute("SELECT * FROM recipes WHERE code=?", (code,)).fetchone()
            if r is None:
                raise LookupError(f"配方不存在: {code}")
            rid = r["id"]
            if version is None:
                rv = c.execute(
                    "SELECT * FROM recipe_versions WHERE recipe_id=? ORDER BY version"
                    " DESC LIMIT 1",
                    (rid,),
                ).fetchone()
            else:
                rv = c.execute(
                    "SELECT * FROM recipe_versions WHERE recipe_id=? AND version=?",
                    (rid, version),
                ).fetchone()
            if rv is None:
                raise LookupError(f"配方版本不存在: {code}@{version}")
            return {
                "recipe_id": rid,
                "recipe_version_id": rv["id"],
                "code": code,
                "name": r["name"],
                "version": rv["version"],
                "created_at": rv["created_at"],
                "message": rv["message"],
                "spec": json.loads(rv["spec_json"]),
            }

    def list_recipes(self) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT r.code,r.name,MAX(rv.version) latest FROM recipes r"
                " JOIN recipe_versions rv ON rv.recipe_id=r.id GROUP BY r.id"
            ).fetchall()
            return [dict(x) for x in rows]

    def list_recipe_versions(self, code: str) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT rv.version,rv.created_at,rv.message FROM recipes r"
                " JOIN recipe_versions rv ON rv.recipe_id=r.id WHERE r.code=?"
                " ORDER BY rv.version",
                (code,),
            ).fetchall()
            return [dict(x) for x in rows]

    # ------------------------------------------------------------- results
    def insert_result(
        self,
        *,
        recipe_code: str,
        recipe_version_id: int,
        library_version_id: int,
        status: str,
        cost: float | None,
        report: dict,
        basis: list[int] | None,
        structure_key: str,
        mode: str,
        job_id: int | None,
        save_warm: bool = True,
    ) -> int:
        with self.connect() as c:
            cur = c.execute(
                "INSERT INTO optimization_results(recipe_code,recipe_version_id,"
                "library_version_id,status,cost,report_json,basis_json,structure_key,"
                "mode,job_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    recipe_code,
                    recipe_version_id,
                    library_version_id,
                    status,
                    cost,
                    json.dumps(report, ensure_ascii=False),
                    json.dumps(basis) if basis is not None else None,
                    structure_key,
                    mode,
                    job_id,
                    time.time(),
                ),
            )
            rid = cur.lastrowid
            if basis is not None and save_warm:
                c.execute(
                    "INSERT INTO warm_basis(recipe_code,structure_key,basis_json,"
                    "library_version_id,updated_at) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(recipe_code,structure_key) DO UPDATE SET"
                    " basis_json=excluded.basis_json,library_version_id=excluded.library_version_id,"
                    "updated_at=excluded.updated_at",
                    (
                        recipe_code,
                        structure_key,
                        json.dumps(basis),
                        library_version_id,
                        time.time(),
                    ),
                )
            return rid

    def get_warm_basis(self, recipe_code: str, structure_key: str):
        with self.connect() as c:
            r = c.execute(
                "SELECT basis_json,library_version_id FROM warm_basis"
                " WHERE recipe_code=? AND structure_key=?",
                (recipe_code, structure_key),
            ).fetchone()
            return None if r is None else (json.loads(r["basis_json"]), r["library_version_id"])

    def commit_warm_bases(self, entries: list[tuple[str, str, list[int], int]]):
        """作业成功结束时统一提交本批热启动基（取消时不调用，自然不留痕）。"""

        now = time.time()
        with self.connect() as c:
            c.executemany(
                "INSERT INTO warm_basis(recipe_code,structure_key,basis_json,"
                "library_version_id,updated_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(recipe_code,structure_key) DO UPDATE SET"
                " basis_json=excluded.basis_json,library_version_id=excluded.library_version_id,"
                "updated_at=excluded.updated_at",
                [
                    (code, skey, json.dumps(basis), lib_id, now)
                    for code, skey, basis, lib_id in entries
                ],
            )

    def get_result(self, result_id: int) -> dict:
        with self.connect() as c:
            r = c.execute(
                "SELECT * FROM optimization_results WHERE id=?", (result_id,)
            ).fetchone()
            if r is None:
                raise LookupError(result_id)
            d = dict(r)
            d["report"] = json.loads(d.pop("report_json"))
            d.pop("basis_json", None)
            return d

    def list_results(self, recipe_code: str, limit: int = 50) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT id,recipe_version_id,library_version_id,status,cost,mode,"
                "job_id,created_at FROM optimization_results WHERE recipe_code=?"
                " ORDER BY id DESC LIMIT ?",
                (recipe_code, limit),
            ).fetchall()
            return [dict(x) for x in rows]

    # ---------------------------------------------------------------- jobs
    def create_job(self, library_version_id: int, recipe_codes: list[str]) -> int:
        """建作业并在**同一事务**内锁定每个配方当前的最新规格版本。

        锁定发生在提交那一刻（与库版本同一时点）：此后改配方、发新库版本
        都不影响本作业；job_items.recipe_version 即锁定凭证，排队中即可查。
        配方代码不存在时版本记 NULL，由 worker 启动时补锁（同老数据）。
        """

        with self.connect() as c:
            now = time.time()
            cur = c.execute(
                "INSERT INTO jobs(library_version_id,status,total,created_at)"
                " VALUES (?,'pending',?,?)",
                (library_version_id, len(recipe_codes), now),
            )
            jid = cur.lastrowid
            rows = []
            for i, code in enumerate(recipe_codes):
                r = c.execute(
                    "SELECT MAX(rv.version) v FROM recipes r"
                    " JOIN recipe_versions rv ON rv.recipe_id=r.id WHERE r.code=?",
                    (code,),
                ).fetchone()
                rows.append((jid, code, i, "pending", r["v"]))
            c.executemany(
                "INSERT INTO job_items(job_id,recipe_code,seq,status,recipe_version)"
                " VALUES (?,?,?,?,?)",
                rows,
            )
            return jid

    def lock_job_item_versions(self, job_id: int) -> dict[str, int | None]:
        """返回本作业 {配方代码: 锁定的规格版本}。

        升级前排队的作业该项为 NULL：在启动时补锁为当时最新版本并落库，
        保证老作业不丢不卡；已锁定的行原样返回。
        """

        with self.connect() as c:
            rows = c.execute(
                "SELECT recipe_code,recipe_version FROM job_items WHERE job_id=?",
                (job_id,),
            ).fetchall()
            locks: dict[str, int | None] = {}
            for row in rows:
                v = row["recipe_version"]
                if v is None:
                    r = c.execute(
                        "SELECT MAX(rv.version) v FROM recipes r"
                        " JOIN recipe_versions rv ON rv.recipe_id=r.id"
                        " WHERE r.code=?",
                        (row["recipe_code"],),
                    ).fetchone()
                    v = r["v"]
                    if v is not None:
                        c.execute(
                            "UPDATE job_items SET recipe_version=?"
                            " WHERE job_id=? AND recipe_code=?",
                            (v, job_id, row["recipe_code"]),
                        )
                locks[row["recipe_code"]] = v
            return locks

    def get_job(self, job_id: int) -> dict | None:
        with self.connect() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if r is None:
                return None
            d = dict(r)
            items = c.execute(
                "SELECT recipe_code,seq,status,recipe_version,result_id,error"
                " FROM job_items WHERE job_id=? ORDER BY seq",
                (job_id,),
            ).fetchall()
            d["items"] = [dict(x) for x in items]
            done = sum(1 for x in d["items"] if x["status"] in ("done", "failed", "skipped"))
            d["processed"] = done
            return d

    def set_job_status(self, job_id: int, status: str, **fields: Any):
        allowed = {"started_at", "finished_at", "error"}
        sets = ["status=?"]
        vals: list[Any] = [status]
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        vals.append(job_id)
        with self.connect() as c:
            c.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", vals)

    def set_item(self, job_id: int, recipe_code: str, status: str, **fields: Any):
        allowed = {"result_id", "error"}
        sets = ["status=?"]
        vals: list[Any] = [status]
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        vals += [job_id, recipe_code]
        with self.connect() as c:
            c.execute(
                f"UPDATE job_items SET {', '.join(sets)} WHERE job_id=? AND recipe_code=?",
                vals,
            )

    def pending_job(self) -> int | None:
        """取最早的待处理作业（单 worker 串行调度）。"""

        with self.connect() as c:
            r = c.execute(
                "SELECT id FROM jobs WHERE status IN ('pending','cancelling')"
                " ORDER BY id LIMIT 1"
            ).fetchone()
            return r["id"] if r else None
