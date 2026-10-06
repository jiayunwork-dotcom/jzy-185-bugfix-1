"""应用配置。数据文件默认落在挂载卷 /data，可用环境变量覆盖。

注意：路径在 Storage 实例化时读取环境变量（而不是模块导入时），
便于测试用 monkeypatch 切换到临时目录。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_DATA_DIR = "/data"


@dataclass(frozen=True)
class Settings:
    http_workers: int = 1  # 单进程异步调度；多副本需另配队列

    @staticmethod
    def data_dir() -> str:
        return os.environ.get("FEED_DATA_DIR", DEFAULT_DATA_DIR)

    @staticmethod
    def db_path() -> str:
        return os.environ.get("FEED_DB_PATH", "")

    def resolved_db_path(self) -> str:
        return self.db_path() or os.path.join(self.data_dir(), "feedmill.db")


settings = Settings()
