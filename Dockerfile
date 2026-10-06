FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FEED_DATA_DIR=/data

WORKDIR /app

# 系统层只需要最小运行时；numpy 走 wheel，无需编译工具链
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 数据落在挂载卷
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 单进程：异步批量调度器在进程内串行执行；水平扩展需把作业队列外置
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
