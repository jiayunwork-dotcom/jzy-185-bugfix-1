"""异步批量重优化调度器。

语义
----
* 提交后立即返回作业号；后台单 worker 串行处理（同一时刻只有一个作业在跑，
  从存储层面保证同一配方不会被两个作业并发优化，结果不会互相覆盖）。
* 作业在**提交时**锁定原料库版本与每个配方的规格版本（见 storage.create_job），
  运行期间即使发布了新版本库、或配方师修改了配方产生新版本规格，作业仍用提交时
  锁定的版本，绝不中途混用。作业详情的每个条目都带锁定的配方版本号，可直接对账。
* 取消：尚未开始的配方标记 skipped；正在跑的那一个跑完（计算不可中断且无外部
  IO，耗时很短），其结果写入临时 job_id 并随后随整批一起删除。
  即“取消的作业不得留下半批结果”：取消提交时把作业状态置 cancelling，
  worker 在每个配方之间检查，最终删除该作业产生的全部结果。
"""

from __future__ import annotations

import asyncio
import time

from .optimizer import OptimizationService
from .storage import Storage


class Scheduler:
    def __init__(self, db: Storage, optimizer: OptimizationService):
        self.db = db
        self.optimizer = optimizer
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._task is not None:
            self._wake.set()
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def submit(self, library_version: int, recipe_codes: list[str]) -> int:
        jid = self.db.create_job(library_version, recipe_codes)
        self._wake.set()
        return jid

    def request_cancel(self, job_id: int) -> bool:
        job = self.db.get_job(job_id)
        if job is None:
            return False
        if job["status"] in ("pending", "running"):
            self.db.set_job_status(job_id, "cancelling")
            self._wake.set()
            return True
        return False

    # ------------------------------------------------------------- worker
    async def _worker(self):
        # 不使用 wait_for：它在“事件先 set、任务后 cancel”的顺序下有取消竞态
        # （任务会带着已完成的 future 复活到下一轮）。sleep 能即时响应取消。
        while True:
            try:
                jid = await asyncio.to_thread(self.db.pending_job)
                if jid is None:
                    if self._wake.is_set():
                        self._wake.clear()
                    await asyncio.sleep(0.1)
                    continue
                await asyncio.to_thread(self._run_job, jid)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(0.5)

    def _affected_recipes(self, new_version: int) -> list[str]:
        """默认目标：所有“引用了受影响原料”的配方（用最新规格判断）。"""

        prev = new_version - 1
        try:
            diff = self.db.diff_library_versions(prev, new_version)
        except LookupError:
            diff = {"changed_ingredients": []}
        changed = set(diff["changed_ingredients"])
        out: list[str] = []
        for r in self.db.list_recipes():
            try:
                rv = self.db.get_recipe_version(r["code"])
            except LookupError:
                continue
            used = {i["code"] for i in rv["spec"]["ingredients"]}
            if not changed or used & changed:
                out.append(r["code"])
        return out

    def _run_job(self, job_id: int):
        job = self.db.get_job(job_id)
        if job is None:
            return
        # 作业可能在 worker 取到它之前就已被取消：不得把 cancelling 改回 running
        pre_cancelled = job["status"] == "cancelling"
        lib_version_id = job["library_version_id"]
        lib = self.db.get_library_version_by_id(lib_version_id)
        # 锁定库版本号
        lib_version = lib["version"]
        if not pre_cancelled:
            self.db.set_job_status(job_id, "running", started_at=time.time())

        items = job["items"]
        cancelled = pre_cancelled
        pending_bases: list[tuple[str, str, list[int], int]] = []
        for item in items:
            current = self.db.get_job(job_id)
            assert current is not None
            if current["status"] == "cancelling":
                cancelled = True
            if cancelled:
                self.db.set_item(job_id, item["recipe_code"], "skipped")
                continue
            self.db.set_item(job_id, item["recipe_code"], "running")
            try:
                # 配方规格版本在提交时已锁定（见 job_items.recipe_version）；
                # 库版本始终用提交时锁定的 lib_version。运行期间改配方不影响本批。
                locked_version = item["recipe_version"]
                if locked_version is None:
                    # 理论上不会发生（建作业时即锁定）；防御性回退并显式报错，
                    # 绝不静默退化为“当前最新版本”。
                    raise RuntimeError(
                        f"作业 {job_id} 的配方 {item['recipe_code']} 缺少锁定版本"
                    )
                report = self.optimizer.optimize(
                    item["recipe_code"],
                    recipe_version=locked_version,
                    lib_version=lib_version,
                    use_warm=True,
                    mode="batch",
                    job_id=job_id,
                    persist=True,
                    save_basis=False,  # 基先不入库，作业成功结束时统一提交
                )
                if report["status"] == "optimal":
                    pending_bases.append(
                        (
                            item["recipe_code"],
                            report["structure_key"],
                            report.pop("_basis"),
                            lib_version_id,
                        )
                    )
                self.db.set_item(
                    job_id,
                    item["recipe_code"],
                    "done",
                    result_id=report.get("result_id"),
                )
            except Exception as exc:  # 单个配方失败不拖垮整批
                self.db.set_item(
                    job_id, item["recipe_code"], "failed", error=str(exc)
                )

        if cancelled:
            # 删除本批写入的全部结果；热启动基从未落库，无需清理
            self._purge_job_results(job_id)
            self.db.set_job_status(job_id, "cancelled", finished_at=time.time())
        else:
            self.db.commit_warm_bases(pending_bases)
            self.db.set_job_status(job_id, "completed", finished_at=time.time())

    def _purge_job_results(self, job_id: int):
        with self.db.connect() as c:
            c.execute(
                "DELETE FROM optimization_results WHERE job_id=?", (job_id,)
            )
