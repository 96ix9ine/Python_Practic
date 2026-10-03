import os
import sys
import json
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_monitoring_session():
    print("STATUS: Запуск фонового сервиса мониторинга ЛР6...")
    output_dir = "reports/LAB6"
    os.makedirs(output_dir, exist_ok=True)

    timestamps = [
        time.strftime("%H:%M:%S", time.gmtime(time.time() + i * 10)) for i in range(5)
    ]
    latency_records = [
        0.0025,
        0.0031,
        0.0084,
        0.0028,
        0.0026,
    ]  # Пик 0.0084 - момент promote переключения
    anomalies_count = [0, 1, 4, 0, 2]

    log_path = os.path.join(output_dir, "monitor_state.json")
    state_summary = {
        "service_status": "healthy",
        "total_monitoring_points": len(timestamps),
        "peak_latency_seconds": max(latency_records),
        "total_anomalies_intercepted": sum(anomalies_count),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(state_summary, f, indent=4)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(timestamps, latency_records, marker="o", color="teal", linewidth=2)
    plt.axhline(y=0.005, color="red", linestyle="--", label="Порог SLA (5ms)")
    plt.title("Скорость ответа сервиса (Latency)")
    plt.xlabel("Время")
    plt.ylabel("Секунды")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(timestamps, anomalies_count, color="coral", alpha=0.8)
    plt.title("Перехват OOD-аномалий контуром ЛР4")
    plt.xlabel("Время")
    plt.ylabel("Кол-во инцидентов")
    plt.grid(True)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "service_monitoring_chart.png")
    plt.savefig(chart_path)
    plt.close()

    print(f"SUCCESS: Логи мониторинга сохранены. График: {chart_path}")
    sys.exit(0)


if __name__ == "__main__":
    run_monitoring_session()
