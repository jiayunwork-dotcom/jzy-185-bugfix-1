"""批量重优化作业接口：提交、查进度、取消。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas import BatchIn

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", status_code=202)
def submit_job(body: BatchIn, request: Request):
    db = request.app.state.storage
    sched = request.app.state.scheduler
    try:
        if body.library_version is None:
            lib = db.get_library()
        else:
            lib = db.get_library(body.library_version)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if body.recipe_codes is None:
        recipe_codes = sched._affected_recipes(lib["version"])
    else:
        missing = [c for c in body.recipe_codes if not _recipe_exists(db, c)]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=[
                    {"field": "recipe_codes", "message": f"配方不存在: {c}"}
                    for c in missing
                ],
            )
        recipe_codes = body.recipe_codes
    if not recipe_codes:
        raise HTTPException(
            status_code=422,
            detail=[{"field": "recipe_codes", "message": "没有需要重优化的配方"}],
        )
    try:
        jid = sched.submit(lib["id"], recipe_codes)
    except LookupError as exc:
        # 提交瞬间逐个锁定配方当前最新版本；配方此刻缺失则拒绝整批
        raise HTTPException(
            status_code=422,
            detail=[{"field": "recipe_codes", "message": str(exc)}],
        )
    return {"job_id": jid, "total": len(recipe_codes), "library_version": lib["version"]}


def _recipe_exists(db, code: str) -> bool:
    try:
        db.get_recipe_version(code)
        return True
    except LookupError:
        return False


@router.get("/{job_id}")
def get_job(job_id: int, request: Request):
    db = request.app.state.storage
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"作业不存在: {job_id}")
    return job


@router.post("/{job_id}/cancel", status_code=200)
def cancel_job(job_id: int, request: Request):
    sched = request.app.state.scheduler
    ok = sched.request_cancel(job_id)
    if not ok:
        raise HTTPException(
            status_code=409, detail="作业不存在或已结束，无法取消"
        )
    return {"job_id": job_id, "status": "cancelling"}
