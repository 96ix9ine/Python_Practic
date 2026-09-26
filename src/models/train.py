import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
import psutil
import hashlib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler

from configs.config_schema import Lab2Config
from src.models.pipeline import DigitsPipeline


def get_memory_usage_mb() -> float:
    """Возвращает текущее потребление памяти процессом в МБ."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def bootstrap_metric(
    y_true, y_pred, metric_func, n_bootstraps=500, random_seed=42, **kwargs
) -> tuple[float, float]:
    """Вычисляет среднее значение метрики и доверительный интервал (бутстрэп)."""
    rng = np.random.default_rng(random_seed)
    bootstrapped_scores = []

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    base_score = metric_func(y_true, y_pred, **kwargs)

    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices], **kwargs)
        bootstrapped_scores.append(score)

    if not bootstrapped_scores:
        return base_score, 0.0

    std_err = np.std(bootstrapped_scores)
    return float(base_score), float(1.96 * std_err)


def main():
    print("STATUS: Старт пайплайна с интеграцией MLflow...")
    try:
        cfg = Lab2Config()
        data_path = cfg.pipeline.data_path

        if not os.path.exists(data_path):
            print(f"CRITICAL ERROR: Файл данных {data_path} не найден!")
            sys.exit(1)

        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("LAB2_Digits_Pipeline")

        np.random.seed(cfg.pipeline.model.random_seed)
        full_df = pd.read_csv(data_path)

        print("STATUS: Проверка негативного контроля (ловушка утечки)...")
        # Обучаем scaler на ВСЕХ данных до разделения выборки
        leakage_scaler = StandardScaler()
        X_leaked = leakage_scaler.fit_transform(full_df.drop(columns=["target"]).values)
        X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
            X_leaked,
            full_df["target"].values,
            test_size=cfg.pipeline.batch_size / len(full_df),
            random_state=cfg.pipeline.model.random_seed,
        )
        from sklearn.linear_model import LogisticRegression

        leaked_model = LogisticRegression(max_iter=100)
        leaked_model.fit(X_tr_l, y_tr_l)
        leakage_score = accuracy_score(y_te_l, leaked_model.predict(X_te_l))
        print(
            f"ALERT [Leakage Trap]: Обнаружена завышенная точность при утечке: {leakage_score:.4f}"
        )

        train_df, test_df = train_test_split(
            full_df,
            test_size=cfg.pipeline.batch_size / len(full_df),
            random_state=cfg.pipeline.model.random_seed,
            stratify=full_df["target"],
        )

        X_test = test_df.drop(columns=["target"]).values
        y_test = test_df["target"].values

        results = {}

        print("STATUS: Оценка наивного ориентира...")
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(train_df.drop(columns=["target"]).values, train_df["target"].values)
        dummy_preds = dummy.predict(X_test)
        dummy_acc, dummy_acc_ci = bootstrap_metric(y_test, dummy_preds, accuracy_score)

        # Открываем прогон в MLflow для фиксации Run ID
        with mlflow.start_run(run_name="Full_Dataset") as run_full:
            print("STATUS: Запуск режима [Полный датасет]...")
            mem_start_full = get_memory_usage_mb()
            time_start_full = time.time()

            X_train_full = train_df.drop(columns=["target"]).values
            y_train_full = train_df["target"].values

            pipe_full = DigitsPipeline(cfg)
            pipe_full.fit_full(X_train_full, y_train_full)

            time_end_full = time.time()
            mem_end_full = get_memory_usage_mb()

            full_preds = pipe_full.predict(X_test)
            full_acc, full_acc_ci = bootstrap_metric(y_test, full_preds, accuracy_score)
            full_prec, full_prec_ci = bootstrap_metric(
                y_test, full_preds, precision_score, average="macro"
            )

            # Логируем параметры и метрики в MLflow для отчета
            mlflow.log_params(cfg.pipeline.model.model_dump())
            mlflow.log_metrics(
                {
                    "accuracy": full_acc,
                    "precision_macro": full_prec,
                    "time_sec": time_end_full - time_start_full,
                    "memory_mb": mem_end_full - mem_start_full,
                }
            )

            results["full"] = {
                "accuracy": f"{full_acc:.4f} ± {full_acc_ci:.4f}",
                "precision_macro": f"{full_prec:.4f} ± {full_prec_ci:.4f}",
                "time_sec": round(time_end_full - time_start_full, 4),
                "memory_mb": round(max(0.0, mem_end_full - mem_start_full), 4),
                "run_id": run_full.info.run_id,
            }

        with mlflow.start_run(run_name="Chunk_Dataset") as run_chunk:
            print("STATUS: Запуск режима [Чанки]...")
            train_df.to_csv("data/raw/train_tmp.csv", index=False)

            mem_start_chunk = get_memory_usage_mb()
            time_start_chunk = time.time()

            pipe_chunk = DigitsPipeline(cfg)

            for chunk in pd.read_csv(
                "data/raw/train_tmp.csv", chunksize=cfg.pipeline.chunk_size
            ):
                X_chunk = chunk.drop(columns=["target"]).values
                y_chunk = chunk["target"].values
                pipe_chunk.partial_fit_chunk(X_chunk, y_chunk)

            time_end_chunk = time.time()
            mem_end_chunk = get_memory_usage_mb()

            if os.path.exists("data/raw/train_tmp.csv"):
                os.remove("data/raw/train_tmp.csv")

            chunk_preds = pipe_chunk.predict(X_test)
            chunk_acc, chunk_acc_ci = bootstrap_metric(
                y_test, chunk_preds, accuracy_score
            )
            chunk_prec, chunk_prec_ci = bootstrap_metric(
                y_test, chunk_preds, precision_score, average="macro"
            )

            # Логируем параметры и метрики чанков в MLflow
            mlflow.log_params(cfg.pipeline.model.model_dump())
            mlflow.log_param("chunk_size", cfg.pipeline.chunk_size)
            mlflow.log_metrics(
                {
                    "accuracy": chunk_acc,
                    "precision_macro": chunk_prec,
                    "time_sec": time_end_chunk - time_start_chunk,
                    "memory_mb": mem_end_chunk - mem_start_chunk,
                }
            )

            results["chunk"] = {
                "accuracy": f"{chunk_acc:.4f} ± {chunk_acc_ci:.4f}",
                "precision_macro": f"{chunk_prec:.4f} ± {chunk_prec_ci:.4f}",
                "time_sec": round(time_end_chunk - time_start_chunk, 4),
                "memory_mb": round(max(0.0, mem_end_chunk - mem_start_chunk), 4),
                "run_id": run_chunk.info.run_id,
            }

        os.makedirs("reports/LAB2", exist_ok=True)

        df_res = pd.DataFrame(results).T
        df_res.to_csv("reports/LAB2/ml_metrics.csv")
        print("\nSTATUS: Сводная таблица метрик успешно сохранена со всеми Run ID:")
        print(df_res)

        model_path = "reports/LAB2/ml_model.joblib"
        joblib.dump(pipe_full, model_path)

        with open(model_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"SUCCESS: Модель сохранена. Хеш: {model_hash}")

        sys.exit(0)

    except Exception as e:
        print(f"CRITICAL ERROR: Сбой в процессе обучения: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
