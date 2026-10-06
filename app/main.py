"""FastAPI 应用入口：组装存储、优化器、异步调度器。"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import jobs, library, optimization, recipes
from .config import settings
from .services.optimizer import OptimizationService
from .services.scheduler import Scheduler
from .services.storage import Storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.data_dir(), exist_ok=True)
    storage = Storage()
    optimizer = OptimizationService(storage)
    scheduler = Scheduler(storage, optimizer)
    app.state.storage = storage
    app.state.optimizer = optimizer
    app.state.scheduler = scheduler
    scheduler.start()
    try:
        yield
    finally:
        await scheduler.stop()


app = FastAPI(title="饲料配方优化服务", version="1.0.0", lifespan=lifespan)

app.include_router(library.router)
app.include_router(recipes.router)
app.include_router(optimization.router)
app.include_router(jobs.router)


@app.get("/health")
def health():
    return {"status": "ok"}
