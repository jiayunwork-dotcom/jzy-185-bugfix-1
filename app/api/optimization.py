"""即时优化、结果历史与版本对比接口。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas import CompareIn, OptimizeIn

router = APIRouter(prefix="/api", tags=["optimization"])


@router.post("/optimize")
def optimize(body: OptimizeIn, request: Request):
    opt = request.app.state.optimizer
    try:
        return opt.optimize(
            body.recipe_code,
            recipe_version=body.recipe_version,
            lib_version=body.library_version,
            use_warm=body.warm_start,
            mode="instant",
            persist=True,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        # build_lp 的带字段名错误
        field = getattr(exc, "field", None)
        if field:
            raise HTTPException(
                status_code=422,
                detail=[{"field": field, "message": getattr(exc, "message", str(exc))}],
            )
        raise


@router.get("/recipes/{code}/results")
def result_history(code: str, request: Request, limit: int = 50):
    db = request.app.state.storage
    return {"recipe_code": code, "results": db.list_results(code, limit)}


@router.get("/results/{result_id}")
def get_result(result_id: int, request: Request):
    db = request.app.state.storage
    try:
        return db.get_result(result_id)
    except LookupError:
        raise HTTPException(status_code=404, detail=f"结果不存在: {result_id}")


@router.post("/compare")
def compare(body: CompareIn, request: Request):
    opt = request.app.state.optimizer
    try:
        return opt.compare(body.result_a, body.result_b)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
