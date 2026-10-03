import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
import mlflow

from configs.config_schema import Lab5Config
from src.data.synthetic_generator import ControlledSyntheticGenerator
from src.utils.metrics_compare import calculate_psi, calculate_ks_statistic


def main():
    print("STATUS: Старт управляемого конвейера генерации данных ЛР5...")
    try:
        cfg = Lab5Config()
        os.makedirs("reports/LAB5", exist_ok=True)

        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("LAB5_Synthetic_Generator")

        real_df = pd.read_csv(cfg.pipeline.data_path)
        X_real = real_df.drop(columns=["target"]).values

        cfg_sim = Lab5Config()
        cfg_sim.pipeline.generator.mode = "similar"
        gen_sim = ControlledSyntheticGenerator(cfg_sim)
        df_sim = gen_sim.generate_dataset()
        X_sim = df_sim.drop(columns=["target"]).values

        cfg_rand = Lab5Config()
        cfg_rand.pipeline.generator.mode = "random"
        gen_rand = ControlledSyntheticGenerator(cfg_rand)
        df_rand = gen_rand.generate_dataset()
        X_rand = df_rand.drop(columns=["target"]).values

        metrics_records = []
        for i in range(64):
            psi_sim = calculate_psi(X_sim[:, i], X_real[:, i])
            ks_sim = calculate_ks_statistic(X_sim[:, i], X_real[:, i])

            psi_rand = calculate_psi(X_rand[:, i], X_real[:, i])
            ks_rand = calculate_ks_statistic(X_rand[:, i], X_real[:, i])

            metrics_records.append(
                {
                    "pixel_index": i,
                    "sim_psi": psi_sim,
                    "sim_ks": ks_sim,
                    "rand_psi": psi_rand,
                    "rand_ks": ks_rand,
                }
            )

        df_metrics = pd.DataFrame(metrics_records)
        df_metrics.to_csv("reports/LAB5/synthetic_metrics.csv", index=False)

        sim_ks_pass_ratio = np.mean(df_metrics["sim_ks"] <= 0.1)
        rand_psi_pass_ratio = np.mean(df_metrics["rand_psi"] >= 0.2)

        config_path = "reports/LAB5/generator_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_sim.pipeline.generator.model_dump(), f, indent=4)

        df_repeat = ControlledSyntheticGenerator(cfg_sim).generate_dataset()

        hash_original = hashlib.sha256(
            df_sim.to_csv(index=False).encode("utf-8")
        ).hexdigest()
        hash_repeat = hashlib.sha256(
            df_repeat.to_csv(index=False).encode("utf-8")
        ).hexdigest()

        with mlflow.start_run(run_name="Synthetic_Similar_Mode") as run:
            mlflow.log_params(cfg_sim.pipeline.generator.model_dump())
            mlflow.log_metrics(
                {
                    "sim_ks_pass_ratio": sim_ks_pass_ratio,
                    "dataset_hash_match": 1.0 if hash_original == hash_repeat else 0.0,
                }
            )
            run_id = run.info.run_id

        print("\n=== ИТОГИ МАТЕМАТИЧЕСКОГО СРАВНЕНИЯ ЛР5 ===")
        print(
            f"Доля пикселей с KS <= 0.1 в режиме 'похожие': {sim_ks_pass_ratio:.4f} (Требуется >= 0.80)"
        )
        print(
            f"Доля пикселей с PSI >= 0.2 в режиме 'случайные': {rand_psi_pass_ratio:.4f} (Требуется >= 0.50)\n"
        )
        print(
            f"Побайтовое совпадение повторного прогона по хэшу: {hash_original == hash_repeat}\n"
        )
        print(f"Хеш синтетической выборки: {hash_original}")
        print(f"Run ID: {run_id}")

        sys.exit(0)
    except Exception as e:
        print(f"CRITICAL ERROR: Сбой конвейера генератора: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
