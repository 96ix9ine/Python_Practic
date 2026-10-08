import os
import sys
import time
import pandas as pd
from fastapi.testclient import TestClient
from src.api.service import app


def run_benchmark():
    print("STATUS: Запуск параллельного бенчмарка инференса joblib (ЛР7)...")
    os.makedirs("reports/LAB7", exist_ok=True)

    single_sample = {"pixels": [0.0] * 64}
    payload = {"batch": [single_sample] * 200}

    results = []

    with TestClient(app) as client:
        print("STATUS: Прогрев пула воркеров...")
        for _ in range(3):
            client.post("/predict/batch?n_jobs=1", json=payload)

        for workers in [1, 2, 4]:
            print(f"STATUS: Бенчмарк для n_jobs={workers}...")
            durations = []

            for repeat in range(3):
                start_time = time.time()
                response = client.post(f"/predict/batch?n_jobs={workers}", json=payload)
                duration = time.time() - start_time

                if response.status_code == 200:
                    durations.append(duration)

            if durations:
                avg_duration = sum(durations) / len(durations)
                throughput = 200 / avg_duration

                results.append(
                    {
                        "workers_used": workers,
                        "avg_latency_seconds": round(avg_duration, 4),
                        "throughput_samples_per_sec": round(throughput, 2),
                    }
                )
            else:
                print(f"ERROR: Запросы для воркеров {workers} упали!")

    df_bench = pd.DataFrame(results)
    df_bench.to_csv("reports/LAB7/parallel_benchmark.csv", index=False)
    print("\n=== РЕЗУЛЬТАТЫ ПАРАЛЛЕЛЬНОГО БЕНЧМАРКА ===")
    print(df_bench)
    print("STATUS: Файл parallel_benchmark.csv успешно сохранен.")


if __name__ == "__main__":
    run_benchmark()
