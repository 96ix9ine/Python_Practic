import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configs.config_schema import Lab1Config


def main():
    cfg = Lab1Config()
    os.makedirs(cfg.eda.output_dir, exist_ok=True)

    # Загрузка данных
    df = pd.read_csv("data/raw/digits.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    # 1. Сбор базовой структуры и пропусков
    stats = X.describe().T
    stats["variance"] = X.var()
    stats["missing_values"] = X.isnull().sum()

    # Считаем аномалии: пиксели со значениями вне диапазона 0-16
    stats["outliers_count"] = ((X < 0) | (X > 16)).sum()

    # Сохраняем основные статистики признаков
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
    # Сохраняем сводку по классам в отдельный файл для машиночитаемости
    class_summary.to_csv(
        os.path.join(cfg.eda.output_dir, "class_distribution.csv"), index=False
    )

    # 3. График: Сетка примеров классов (Требование P2)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.ravel()
    for i in range(10):
        img_matrix = X[y == i].iloc[0].values.reshape(8, 8)
        axes[i].imshow(img_matrix, cmap="gray")
        axes[i].set_title(f"Класс {i}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.eda.output_dir, "digits_grid.png"))
    plt.close()

    # 4. График проблемы: Избыточная размерность / Затухание дисперсии
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

    # Числовая подпись прямо на графике
    plt.text(
        12,
        float(variances.max() / 2),
        f"Всего признаков: 64\nПикселей с Var=0: {zero_var_count}\nПропусков: 0",
        bbox=dict(facecolor="orange", alpha=0.5),
    )

    plt.savefig(os.path.join(cfg.eda.output_dir, "problem_dimension.png"))
    plt.close()
    print("EDA успешно выполнен")


if __name__ == "__main__":
    main()
