import os
import sys
import json
import pandas as pd
import numpy as np

from evidently import Report
from evidently.presets import DataDriftPreset

from src.data.synthetic_generator import ControlledSyntheticGenerator
from configs.config_schema import Lab5Config


def run_drift_monitoring():
    print("STATUS: Запуск контура Evidently-мониторинга потока данных (ЛР7)...")
    output_dir = "reports/LAB7"
    os.makedirs(output_dir, exist_ok=True)

    cfg = Lab5Config()
    real_df = pd.read_csv(cfg.pipeline.data_path).drop(columns=["target"])

    monitored_pixels = ["pixel_28", "pixel_36", "pixel_44"]

    # Окна 1 и 2 - синтетика (similar)
    cfg.pipeline.generator.num_samples = 200
    cfg.pipeline.generator.mode = "similar"
    clean_stream = (
        ControlledSyntheticGenerator(cfg).generate_dataset().drop(columns=["target"])
    )

    # Окна 3, 4 и 5 - жесткий дрейф
    cfg.pipeline.generator.num_samples = 300
    cfg.pipeline.generator.drift_flip_labels = True
    drift_stream = (
        ControlledSyntheticGenerator(cfg).generate_dataset().drop(columns=["target"])
    )

    full_stream = pd.concat([clean_stream, drift_stream], ignore_index=True)

    drift_records = []
    alert_window = -1
    window_size = 100

    report = Report([DataDriftPreset(columns=monitored_pixels)])

    print("STATUS: Имитация скользящих окон на потоке...")
    for w in range(5):
        current_window = full_stream.iloc[w * window_size : (w + 1) * window_size]
        stream_type = "clean" if w < 2 else "drifted"

        my_eval = report.run(
            current_window[monitored_pixels], real_df[monitored_pixels]
        )

        eval_json = json.loads(my_eval.json())

        metrics = eval_json.get("metrics", [])
        drift_share = 0.0
        for metric in metrics:
            if "DatasetDriftMetric" in metric.get(
                "metric_name", ""
            ) or "DriftedColumnsCount" in metric.get("metric_name", ""):
                drift_share = metric.get("value", {}).get("share", 0.0)
                break

        if drift_share == 0.0 and metrics:
            drift_share = float(eval_json["metrics"][0]["value"]["share"])

        is_alert = bool(drift_share >= 0.3)

        if is_alert and alert_window == -1:
            alert_window = w + 1

        drift_records.append(
            {
                "window_number": w + 1,
                "stream_type": stream_type,
                "drifted_features_share": round(drift_share, 4),
                "alert_triggered": int(is_alert),
            }
        )

    df_drift = pd.DataFrame(drift_records)
    df_drift.to_csv(os.path.join(output_dir, "drift_metrics.csv"), index=False)

    my_eval.save_html(os.path.join(output_dir, "evidently_drift_report.html"))

    delay = alert_window - 3 if alert_window != -1 else -1

    summary = {
        "window_size": window_size,
        "total_windows": 5,
        "drift_detected": bool(alert_window != -1),
        "alert_triggered_window": alert_window,
        "detection_delay_windows": delay,
        "false_alerts_on_clean_stream": int(df_drift.iloc[:2]["alert_triggered"].sum()),
    }
    with open(
        os.path.join(output_dir, "drift_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n=== ИТОГИ ДРИЛЛОВ ДРЕЙФА ДАННЫХ ===")
    print(df_drift)
    print(f"\nSTATUS: Дрейф успешно обнаружен на окне: {alert_window}")
    print(f"STATUS: Задержка детекции: {delay} окон")
    sys.exit(0)


if __name__ == "__main__":
    run_drift_monitoring()
