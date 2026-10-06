"""配方规格接口（每次修改产生新版本）。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas import RecipeIn
from ..services.validation import ValidationErrors, validate_recipe

router = APIRouter(prefix="/api/recipes", tags=["recipes"])


@router.get("")
def list_recipes(request: Request):
    db = request.app.state.storage
    return {"recipes": db.list_recipes()}


@router.get("/{code}/versions")
def recipe_versions(code: str, request: Request):
    db = request.app.state.storage
    try:
        return {"code": code, "versions": db.list_recipe_versions(code)}
    except Exception:
        raise HTTPException(status_code=404, detail=f"配方不存在: {code}")


@router.get("/{code}")
def get_recipe(code: str, request: Request, version: int | None = None):
    db = request.app.state.storage
    try:
        return db.get_recipe_version(code, version)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.put("/{code}", status_code=201)
def upsert_recipe(code: str, body: RecipeIn, request: Request):
    db = request.app.state.storage
    if body.code != code:
        raise HTTPException(
            status_code=422,
            detail=[{"field": "code", "message": "路径与请求体中的配方代码不一致"}],
        )
    # 用最新发布库的原料集合校验引用
    try:
        lib = db.get_library()
        ingredient_codes = set(lib["ingredients"])
    except LookupError:
        ingredient_codes = set()
    known = db.nutrient_codes()
    try:
        validate_recipe(body, known, ingredient_codes)
    except ValidationErrors as exc:
        raise HTTPException(status_code=422, detail=exc.errors)
    version = db.create_recipe(code, body.name, body.model_dump(), body.message)
    return {"code": code, "version": version}
