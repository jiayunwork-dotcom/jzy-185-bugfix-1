"""Pydantic 请求/响应模型（HTTP 边界）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------- ingredients
class IngredientIn(BaseModel):
    code: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price: float
    nutrients: dict[str, float] = Field(default_factory=dict)


class NutrientIn(BaseModel):
    code: str = Field(min_length=1)
    name: str = Field(min_length=1)
    unit: str = ""


class PublishLibraryIn(BaseModel):
    ingredients: list[IngredientIn]
    message: str = ""


# ----------------------------------------------------------------- recipe
class IngredientLimitIn(BaseModel):
    code: str
    min_ratio: float = 0.0
    max_ratio: float = 1.0


class NutrientBoundIn(BaseModel):
    code: str
    lower: float | None = None
    upper: float | None = None


class RatioIn(BaseModel):
    numerator: str
    denominator: str
    lower: float | None = None
    upper: float | None = None


class RecipeIn(BaseModel):
    code: str = Field(min_length=1)
    name: str = Field(min_length=1)
    ingredients: list[IngredientLimitIn]
    nutrients: list[NutrientBoundIn] = []
    ratios: list[RatioIn] = []
    total_mass: float = 1.0
    message: str = ""


class OptimizeIn(BaseModel):
    recipe_code: str
    recipe_version: int | None = None
    library_version: int | None = None
    warm_start: bool = True


class BatchIn(BaseModel):
    library_version: int | None = None  # 默认最新
    recipe_codes: list[str] | None = None  # 默认所有引用了受影响原料的配方


class CompareIn(BaseModel):
    recipe_code: str
    result_a: int
    result_b: int
