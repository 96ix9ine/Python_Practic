import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configs.config_schema import Lab1Config


def main():
    cfg = Lab1Config()
    df = pd.read_csv("data/raw/digits.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    # 1. Машиночитаемые сводки (eda_stats.csv)
    stats = X.describe().T
    stats["variance"] = X.var()
    stats.to_csv(os.path.join(cfg.eda.output_dir, cfg.eda.stats_filename))

    # 2. График: Сетка примеров классов (Требование P2)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.ravel()
    for i in range(10):
        # Берем первый образец изображения для каждого класса 0..9
        img_matrix = X[y == i].iloc[0].values.reshape(8, 8)
        axes[i].imshow(img_matrix, cmap="gray")
        axes[i].set_title(f"Класс {i}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.eda.output_dir, "digits_grid.png"))
    plt.close()

    # 3. График проблемы: Избыточная размерность / Затухание дисперсии
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

    # Числовая подпись прямо на графике (Критерий выполнения)
    plt.text(
        15,
        vars(X)["variance"].max() / 2,
        f"Всего признаков: 64\nПикселей с нулевой дисперсией: {zero_var_count}",
        bbox=dict(facecolor="orange", alpha=0.5),
    )

    plt.savefig(os.path.join(cfg.eda.output_dir, "problem_dimension.png"))
    plt.close()
    print("EDA успешно выполнен. Сводки и графики сохранены.")


if __name__ == "__main__":
    main()
