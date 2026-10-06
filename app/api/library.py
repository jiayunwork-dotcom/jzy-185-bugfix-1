"""原料库与营养项管理接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas import NutrientIn, PublishLibraryIn
from ..services.validation import ValidationErrors, validate_ingredients

router = APIRouter(prefix="/api", tags=["library"])


def _services(request: Request):
    return request.app.state.storage, request.app.state.optimizer


@router.get("/nutrients")
def list_nutrients(request: Request):
    db, _ = _services(request)
    return {"nutrients": db.list_nutrients()}


@router.post("/nutrients", status_code=201)
def add_nutrient(body: NutrientIn, request: Request):
    db, _ = _services(request)
    try:
        return db.add_nutrient(body.code, body.name, body.unit)
    except Exception as exc:  # 重复代码
        raise HTTPException(status_code=409, detail=[{"field": "code", "message": str(exc)}])


@router.get("/library/versions")
def list_versions(request: Request):
    db, _ = _services(request)
    return {"versions": db.list_library_versions()}


@router.get("/library")
def get_library(request: Request, version: int | None = None):
    db, _ = _services(request)
    try:
        return db.get_library(version)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/library/publish", status_code=201)
def publish_library(body: PublishLibraryIn, request: Request):
    db, _ = _services(request)
    known = db.nutrient_codes()
    try:
        validate_ingredients(body.ingredients, known)
    except ValidationErrors as exc:
        raise HTTPException(status_code=422, detail=exc.errors)
    ingredients = [i.model_dump() for i in body.ingredients]
    version = db.publish_library(ingredients, body.message)
    return {"version": version, "ingredient_count": len(ingredients)}


@router.get("/library/diff")
def diff_versions(request: Request, old: int, new: int):
    db, _ = _services(request)
    try:
        return db.diff_library_versions(old, new)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
