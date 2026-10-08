import os
import sys
import json
import hashlib
import threading
import time
from typing import Any
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
import uvicorn

import multiprocessing
from joblib import Parallel, delayed
from src.schemas.api_schemas import BatchPredictRequest, BatchPredictResponse

from src.schemas.api_schemas import PredictRequest, PredictResponse
from src.models.validator import ImageOODValidator
from configs.config_schema import Lab4Config

app = FastAPI(title="Digits Classification Service (Profile P2)")

REGISTRY_PATH = "reports/LAB6/registry.json"
model_lock = threading.Lock()

MAX_WORKERS = multiprocessing.cpu_count()

active_version_name: str = ""
active_model: Any = None
active_validator: Any = None

metrics_storage = {
    "total_requests": 0,
    "successful_requests": 0,
    "anomaly_detected": 0,
    "latency_sum": 0.0,
}


def calculate_file_sha256(filepath: str) -> str:
    """Вычисление контрольной суммы файла для верификации версий."""
    if not os.path.exists(filepath):
        return ""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_model_by_version(version_name: str) -> bool:
    """Потокобезопасная атомарная загрузка модели и валидатора по версии из реестра."""
    global active_version_name, active_model, active_validator

    if not os.path.exists(REGISTRY_PATH):
        return False

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    target_info = next(
        (v for v in registry["versions"] if v["version"] == version_name), None
    )
    if not target_info:
        return False

    actual_hash = calculate_file_sha256(target_info["path"])
    if actual_hash != target_info["hash"]:
        print(
            f"CRITICAL: The hash of file {target_info['path']} does not match the registry!"
        )
        return False

    try:
        new_model = joblib.load(target_info["path"])

        cfg = Lab4Config()
        new_validator = ImageOODValidator(cfg)
        new_validator.load_params("reports/LAB4/validator_params.npz")

        with model_lock:
            active_version_name = version_name
            active_model = new_model
            active_validator = new_validator

        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Детерминированный жизненный цикл управления весами моделей"""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            reg = json.load(f)
        load_model_by_version(reg["active_version"])
    yield
    with model_lock:
        global active_model, active_validator
        active_model = None
        active_validator = None


app = FastAPI(title="Digits Classification Service (Profile P2)", lifespan=app_lifespan)


@app.get("/health")
def health_check():
    """Проверка доступности сервиса"""
    if active_model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")
    return {"status": "healthy", "active_version": active_version_name}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Инференс модели с обязательным флагом аномалии по профилю P2"""
    global metrics_storage
    start_time = time.time()
    metrics_storage["total_requests"] += 1

    with model_lock:
        local_model = active_model
        local_validator = active_validator
        local_version = active_version_name

    if local_model is None:
        raise HTTPException(
            status_code=503, detail="Service Model is locked or initializing"
        )

    X_input = np.array(request.pixels).reshape(1, -1)

    is_anomaly = bool(local_validator.predict_rejection(X_input)[0] == 1)
    if is_anomaly:
        metrics_storage["anomaly_detected"] += 1

    pred_class = int(local_model.predict(X_input)[0])

    latency = time.time() - start_time
    metrics_storage["latency_sum"] += latency
    metrics_storage["successful_requests"] += 1

    return PredictResponse(
        prediction=pred_class, is_anomaly=is_anomaly, model_version=local_version
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest, n_jobs: int = 2):
    """Пакетный инференс, обрабатывающий пачку запросов параллельно средствами joblib"""
    start_time = time.time()

    workers = min(max(1, n_jobs), MAX_WORKERS)

    with model_lock:
        local_model = active_model
        local_validator = active_validator
        local_version = active_version_name

    if local_model is None:
        raise HTTPException(status_code=503, detail="Model is locked or initializing")

    def process_single_sample(req_item):
        try:
            X_single = np.array(req_item.pixels).reshape(1, -1)
            is_anomaly = (
                bool(local_validator.predict_rejection(X_real=X_single) == 1)
                if local_validator
                else False
            )
            pred_class = int(local_model.predict(X_single))
            return PredictResponse(
                prediction=pred_class,
                is_anomaly=is_anomaly,
                model_version=local_version,
            )
        except Exception as e:
            return PredictResponse(
                prediction=-1, is_anomaly=True, model_version="error_worker"
            )

    parallel_results = Parallel(n_jobs=workers, backend="threading")(
        delayed(process_single_sample)(item) for item in request.batch
    )

    exec_time = time.time() - start_time
    return BatchPredictResponse(
        results=parallel_results,
        processing_time_seconds=round(exec_time, 4),
        workers_used=workers,
    )


@app.get("/metrics")
def get_metrics():
    """Эндпоинт текстового логирования метрик для сервиса мониторинга"""
    avg_latency = 0.0
    if metrics_storage["successful_requests"] > 0:
        avg_latency = (
            metrics_storage["latency_sum"] / metrics_storage["successful_requests"]
        )

    return {
        "total_requests": metrics_storage["total_requests"],
        "successful_requests": metrics_storage["successful_requests"],
        "anomaly_detected": metrics_storage["anomaly_detected"],
        "avg_latency_seconds": round(avg_latency, 6),
    }


@app.get("/models")
def list_models():
    """Список зарегистрированных версий моделей со статусами и хешами"""
    if not os.path.exists(REGISTRY_PATH):
        raise HTTPException(status_code=500, detail="Registry file not found")
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/promote")
def promote_model(version: str):
    """Бесшовное продвижение версии на роль активной"""
    if not os.path.exists(REGISTRY_PATH):
        raise HTTPException(status_code=500, detail="Registry file not found")

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    target = next((v for v in registry["versions"] if v["version"] == version), None)
    if not target:
        raise HTTPException(status_code=44, detail=f"Version {version} not found")

    success = load_model_by_version(version)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Model activation failed (Hash mismatch or file error)",
        )

    for v in registry["versions"]:
        if v["version"] == version:
            v["status"] = "рабочая"
        elif v["status"] == "рабочая":
            v["status"] = "снята"

    registry["active_version"] = version
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4, ensure_ascii=False)

    return {
        "message": f"Successfully promoted version {version} to active",
        "status": "success",
    }


@app.post("/rollback")
def rollback_model(version: str):
    """Откат версии под нагрузкой без простоя"""
    return promote_model(version)


def main():
    """Точка входа для запуска ASGI-сервера uvicorn через CLI-команду"""
    uvicorn.run("src.api.service:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
