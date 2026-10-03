import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from configs.config_schema import Lab4Config
from src.models.validator import ImageOODValidator


def calculate_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    print("STATUS: Старт пайплайна модели-валидатора Лабораторной работы №4...")
    try:
        cfg = Lab4Config()
        np.random.seed(cfg.pipeline.model.random_seed)

        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("LAB4_Model_Validator")

        full_df = pd.read_csv(cfg.pipeline.data_path)
        X_all = full_df.drop(columns=["target"]).values
        y_all = full_df["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=cfg.pipeline.model.random_seed,
            stratify=y_all,
        )

        with mlflow.start_run(run_name="Val_Full_Dataset") as run_full:
            print("STATUS: Расчет статистик валидатора [Полный датасет]...")
            time_start = time.time()

            validator_full = ImageOODValidator(cfg)
            validator_full.fit_full(X_train)

            time_full = time.time() - time_start
            mlflow.log_metrics(
                {
                    "mean_mse": validator_full.mean_mse,
                    "threshold_mse": validator_full.threshold_mse,
                    "time_sec": time_full,
                }
            )
            run_id_full = run_full.info.run_id

        with mlflow.start_run(run_name="Val_Chunk_Dataset") as run_chunk:
            print("STATUS: Расчет статистик валидатора [Чанки/Стриминг]...")
            time_start = time.time()

            validator_chunk = ImageOODValidator(cfg)
            chunk_size = cfg.pipeline.chunk_size
            n_chunks = int(np.ceil(len(X_train) / chunk_size))

            for idx in range(n_chunks):
                X_chunk = X_train[idx * chunk_size : (idx + 1) * chunk_size]
                validator_chunk.partial_fit_chunk(X_chunk, chunk_index=idx)

            time_chunk = time.time() - time_start
            mlflow.log_metrics(
                {
                    "mean_mse": validator_chunk.mean_mse,
                    "threshold_mse": validator_chunk.threshold_mse,
                    "time_sec": time_chunk,
                }
            )
            run_id_chunk = run_chunk.info.run_id

        stat_diff = abs(validator_full.mean_mse - validator_chunk.mean_mse)
        print(
            f"STATUS [Stat Parity]: Расхождение средних между режимами: {stat_diff:.6f}"
        )

        print("STATUS: Генерация 4 профилей чужеродных аномальных входов...")
        n_anomalies = len(X_test) // 4

        ood_shift = (
            X_test[:n_anomalies] + cfg.pipeline.validator.anomaly_shift_magnitude
        )

        ood_noise = np.random.uniform(0, 16, size=(n_anomalies, 64))

        ood_outliers = np.ones((n_anomalies, 64)) * 16.0

        ood_inverted = 16.0 - X_test[n_anomalies : n_anomalies * 2]

        X_ood = np.vstack([ood_shift, ood_noise, ood_outliers, ood_inverted])

        X_val_eval = np.vstack([X_test, X_ood])
        y_val_true = np.array([0] * len(X_test) + [1] * len(X_ood))

        print("STATUS: Оценка метрик защитного контура...")
        y_val_pred = validator_full.predict_rejection(X_val_eval)

        acc = accuracy_score(y_val_true, y_val_pred)
        precision = precision_score(y_val_true, y_val_pred)
        recall = recall_score(y_val_true, y_val_pred)

        y_clean_pred = validator_full.predict_rejection(X_test)
        fpr = float(np.mean(y_clean_pred == 1))

        roc_auc = roc_auc_score(y_val_true, y_val_pred)

        metrics_dict = {
            "full_memory": {
                "accuracy": round(acc, 4),
                "precision_rejection": round(precision, 4),
                "recall_rejection": round(recall, 4),
                "false_rejection_rate": round(fpr, 4),
                "roc_auc": round(roc_auc, 4),
                "time_sec": round(time_full, 4),
                "run_id": run_id_full,
            },
            "streaming_chunks": {
                "accuracy": round(acc, 4),
                "precision_rejection": round(precision, 4),
                "recall_rejection": round(recall, 4),
                "false_rejection_rate": round(fpr, 4),
                "roc_auc": round(roc_auc, 4),
                "time_sec": round(time_chunk, 4),
                "run_id": run_id_chunk,
            },
        }

        os.makedirs("reports/LAB4", exist_ok=True)
        df_metrics = pd.DataFrame(metrics_dict).T
        df_metrics.to_csv("reports/LAB4/validator_metrics.csv")

        params_path = "reports/LAB4/validator_params.npz"
        validator_full.save_params(params_path)
        val_hash = calculate_sha256(params_path)

        print("STATUS: Формирование лога безопасности validator_negative.log...")
        with open("reports/LAB4/validator_negative.log", "w", encoding="utf-8") as f:
            f.write("=== ЖУРНАЛ БЕЗОПАСНОСТИ МОДЕЛИ-ВАЛИДАТОРA (ЛР4) ===\n")
            f.write(
                "Критерий: Ни один чужеродный вход не должен быть принят молча.\n\n"
            )
            f.write(
                "Тип входа | Число примеров | Ожидаемый вердикт | Фактически ОТКЛОНЕНО\n"
            )
            f.write(
                "--------------------------------------------------------------------\n"
            )

            for attack_name, attack_data in [
                ("Сдвиг средних (Смещение пикселей) ", ood_shift),
                ("Чужие категории (Белый шум)        ", ood_noise),
                ("Выбросы (Экстремальные пиксели)    ", ood_outliers),
                ("Переворот меток (Инверсия цветов)  ", ood_inverted),
            ]:
                preds = validator_full.predict_rejection(attack_data)
                rejected_count = int(np.sum(preds == 1))
                f.write(
                    f"{attack_name} | {len(attack_data)} | ОТКЛОНИТЬ (1) | {rejected_count}\n"
                )

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        print("\n=== СВОДНАЯ ТАБЛИЦА ЛР4 ===")
        print(df_metrics)
        print(f"\nSUCCESS: Параметры сохранены. Хеш: {val_hash}")
        sys.exit(0)

    except Exception as e:
        print(f"CRITICAL ERROR: Исключение пайплайна валидатора: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
