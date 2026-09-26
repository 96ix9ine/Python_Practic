import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configs.config_schema import Lab1Config


def main():
    print("STATUS: Запуск разведочного анализа данных (EDA)...")

    try:
        cfg = Lab1Config()

        # Проверяем физическое наличие файла перед чтением
        csv_path = "data/raw/digits.csv"
        if not os.path.exists(csv_path):
            print(f"CRITICAL ERROR: Файл данных не найден по пути {csv_path}!")
            print("STATUS: Запустите сначала скрипт src.data.download_data")
            sys.exit(1)

        # Чтение данных
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["target"])
        y = df["target"]

        # Проверяем структуру
        if X.empty or len(y) == 0:
            print("CRITICAL ERROR: Загруженный датасет пуст или поврежден!")
            sys.exit(1)

        # 1. Расчет статистик
        stats = X.describe().T
        stats["variance"] = X.var()
        stats["missing_values"] = X.isnull().sum()
        stats["outliers_count"] = ((X < 0) | (X > 16)).sum()
        stats.to_csv(os.path.join(cfg.eda.output_dir, cfg.eda.stats_filename))

        # 2. Расчет долей классов
        class_counts = y.value_counts().sort_index()
        class_proportions = y.value_counts(normalize=True).sort_index()
        class_summary = pd.DataFrame(
            {
                "class_label": class_counts.index,
                "absolute_count": class_counts.values,
                "proportion": class_proportions.values,
            }
        )
        class_summary.to_csv(
            os.path.join(cfg.eda.output_dir, "class_distribution.csv"), index=False
        )

        # 3. Генерация графиков
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        axes = axes.ravel()
        for i in range(10):
            # Исправлено: берем первую доступную строку .iloc[0] для класса i
            img_matrix = X[y == i].iloc[0].values.reshape(8, 8)
            axes[i].imshow(img_matrix, cmap="gray")
            axes[i].set_title(f"Класс {i}")
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.eda.output_dir, "digits_grid.png"))
        plt.close()

        # 4. График проблемы
        variances = X.var().sort_values(ascending=False).values
        zero_var_count = int(np.sum(variances == 0.0))

        plt.figure(figsize=(8, 5))
        plt.plot(variances, marker="o", color="darkblue", linewidth=2)
        plt.axhline(
            y=cfg.eda.low_variance_threshold,
            color="red",
            linestyle="--",
            label=f"Порог малоинформативности ({cfg.eda.low_variance_threshold})",
        )
        plt.title("Проблема варианта: Проклятие размерности и затухание дисперсии")
        plt.xlabel("Ранг признака (отсортирован по убыванию дисперсии)")
        plt.ylabel("Дисперсия пикселя")
        plt.grid(True)
        plt.legend()

        plt.text(
            12,
            float(variances.max() / 2),
            f"Всего признаков: 64\nПикселей с Var=0: {zero_var_count}\nПропусков: 0",
            bbox=dict(facecolor="orange", alpha=0.5),
        )

        plt.savefig(os.path.join(cfg.eda.output_dir, "problem_dimension.png"))
        plt.close()

        print(
            "SUCCESS: Разведочный анализ (EDA) успешно завершен. Все артефакты сохранены."
        )
        sys.exit(0)

    except Exception as e:
        print(f"CRITICAL ERROR: Непредвиденный сбой при выполнении EDA: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
