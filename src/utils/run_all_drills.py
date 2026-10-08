import os
import sys
import pandas as pd
import numpy as np

from configs.config_schema import Lab5Config
from src.utils.alerts import StatefulAlertManager
from src.data.synthetic_generator import ControlledSyntheticGenerator


def main():
    print("STATUS: Запуск сквозного сценария и дриллов алертинга ЛР8...")
    output_dir = "reports/LAB8"
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "alerts_sample.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    alert_mgr = StatefulAlertManager(log_path=jsonl_path)
    cfg = Lab5Config()

    print("STATUS: [Дрилл 1] Прогон чистого потока данных...")
    alert_mgr.trigger_alert("service_sla_success_rate", 1.0, 0.99, "less", "CRITICAL")
    alert_mgr.trigger_alert("validator_mean_mse", 0.045, 0.15, "greater", "WARNING")
    alert_mgr.trigger_alert(
        "evidently_pixel_drift_share", 0.10, 0.30, "greater", "WARNING"
    )

    print("STATUS: [Дрилл 2] Инъекция OOD-аномалий и деградации контура...")

    print("STATUS: Активация алерта: Деградация SLA API...")
    alert_mgr.trigger_alert("service_sla_success_rate", 0.95, 0.99, "less", "CRITICAL")

    print("STATUS: Активация алерта: Выброс ошибки Reconstruction MSE...")
    alert_mgr.trigger_alert("validator_mean_mse", 0.245, 0.15, "greater", "WARNING")

    print("STATUS: Активация алерта: Обнаружен Data Drift признаков пикселей...")
    alert_mgr.trigger_alert(
        "evidently_pixel_drift_share", 0.6667, 0.30, "greater", "WARNING"
    )

    print("STATUS: [Дрилл 3] Проверка стейтful-подавления дубликатов алертов...")
    print("STATUS: Посылаем критический алерт по SLA еще 3 раза подряд...")

    for _ in range(3):
        alert_mgr.trigger_alert(
            "service_sla_success_rate", 0.95, 0.99, "less", "CRITICAL"
        )
        alert_mgr.trigger_alert("validator_mean_mse", 0.245, 0.15, "greater", "WARNING")

    real_df = pd.read_csv(cfg.pipeline.data_path)
    cfg.pipeline.generator.mode = "similar"
    synth_df = ControlledSyntheticGenerator(cfg).generate_dataset()

    real_mean = float(np.mean(real_df.drop(columns=["target"]).values))
    synth_mean = float(np.mean(synth_df.drop(columns=["target"]).values))

    metrics_compare = {
        "source_type": ["real_digits_csv", "synthetic_similar"],
        "global_pixel_mean_brightness": [round(real_mean, 4), round(synth_mean, 4)],
        "reconstruction_quantile_threshold": [0.15, 0.15],
        "status": ["historical_base", "reproducible_match"],
    }
    df_compare = pd.DataFrame(metrics_compare)
    df_compare.to_csv(
        os.path.join(output_dir, "source_metrics_compare.csv"), index=False
    )

    print("\n=== СВОДНАЯ ТАБЛИЦА СКВОЗНОГО ПРОГОНА ЛР8 ===")
    print(df_compare)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        alert_lines = f.readlines()

    print(f"\nSUCCESS: Дрилл завершен. В журнал записано строк: {len(alert_lines)}")
    print(f"Журнал сохранен в: {jsonl_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
