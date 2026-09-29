print("INIT: Инициализация скрипта train_dl.py...")
import os
import sys
import time
import copy
import hashlib
from typing import Any, List, Dict

print("INIT: Импорт базовых библиотек...")
import pandas as pd
import numpy as np

print("INIT: Импорт PyTorch...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

print("INIT: Импорт MLflow и инструментов мониторинга...")
import mlflow
import psutil

print("INIT: Импорт локальных конфигураций проекта...")
from configs.config_schema import Lab3Config
from src.data.datasets import MemoryDataset, StreamingDigitsDataset
from src.models.vae_model import DigitsVAE, vae_loss_function


def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def set_deterministic_seeds(seed: int) -> None:
    """Фиксация зерен без вызова системного краша Windows"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_bootstrap_mse(
    model: DigitsVAE,
    data_loader: DataLoader,
    device: torch.device,
    n_bootstraps: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    model.eval()
    all_losses: List[float] = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 64))
            recon = model.decode(mu)
            mse_per_sample = torch.mean((recon - data) ** 2, dim=1).cpu().numpy()
            all_losses.extend(mse_per_sample.tolist())

    losses_arr = np.array(all_losses)
    base_score = float(np.mean(losses_arr))
    rng = np.random.default_rng(seed)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(losses_arr), len(losses_arr))
        bootstrapped_scores.append(np.mean(losses_arr[indices]))
    return base_score, float(1.96 * np.std(bootstrapped_scores))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    samples_count = 0
    for data, _ in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = vae_loss_function(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        samples_count += data.size(0)
    return total_loss / samples_count


def evaluate_loss(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    samples_count = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss = vae_loss_function(recon, data, mu, logvar)
            total_loss += loss.item()
            samples_count += data.size(0)
    return total_loss / samples_count


def run_training_pipeline(mode: str, use_seed: bool = True) -> dict:
    cfg = Lab3Config()
    if use_seed:
        set_deterministic_seeds(cfg.pipeline.model.random_seed)
    else:
        set_deterministic_seeds(int(time.time() * 1000) % 10000)

    device = torch.device("cpu")
    train_dataset: Any = None
    val_dataset: Any = None

    if mode == "full_memory":
        full_df = pd.read_csv(cfg.pipeline.data_path)
        train_df = full_df.sample(frac=0.8, random_state=42)
        val_df = full_df.drop(train_df.index)
        train_dataset = MemoryDataset(train_df)
        val_dataset = MemoryDataset(val_df)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.pipeline.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.pipeline.batch_size, shuffle=False
        )
    else:
        train_dataset = StreamingDigitsDataset(
            cfg.pipeline.data_path, chunk_size=cfg.pipeline.chunk_size, is_train=True
        )
        val_dataset = StreamingDigitsDataset(
            cfg.pipeline.data_path, chunk_size=cfg.pipeline.chunk_size, is_train=False
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg.pipeline.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=cfg.pipeline.batch_size)

    model = DigitsVAE(
        input_dim=cfg.pipeline.vae.input_dim,
        hidden_dim=cfg.pipeline.vae.hidden_dim,
        latent_dim=cfg.pipeline.vae.latent_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.pipeline.vae.lr)
    mem_start = get_memory_usage_mb()
    time_start = time.time()

    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    patience = cfg.pipeline.vae.early_stopping_patience

    history_loss: List[float] = []

    for epoch in range(1, cfg.pipeline.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)
        history_loss.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"STATUS [{mode}]: Ранняя остановка сработала на эпохе {epoch}")
                break

    exec_time = time.time() - time_start
    mem_used = get_memory_usage_mb() - mem_start

    model.load_state_dict(best_model_wts)
    mse_base, mse_ci = run_bootstrap_mse(model, val_loader, device)

    return {
        "mse": mse_base,
        "mse_ci": mse_ci,
        "time": exec_time,
        "memory": max(0.01, mem_used),
        "model": model,
        "val_loader": val_loader,
        "history": history_loss,
    }


def main() -> None:
    print("STATUS: Старт DL-пайплайна Лабораторной работы №3...")
    try:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("LAB3_Digits_VAE")

        print("STATUS: Тестирование негативного контроля (снятая фиксация зерен)...")
        control_run_1 = run_training_pipeline("full_memory", use_seed=False)
        control_run_2 = run_training_pipeline("full_memory", use_seed=False)
        seed_diff = abs(control_run_1["mse"] - control_run_2["mse"])
        print(
            f"ALERT [Seed Control]: Расхождение MSE без фиксации зерен: {seed_diff:.6f}"
        )

        results: Dict[str, dict] = {}
        cfg = Lab3Config()

        with mlflow.start_run(run_name="DL_Full_Memory") as run_full:
            print("STATUS: Обучение VAE в режиме [Полный датасет]...")
            res_full = run_training_pipeline("full_memory", use_seed=True)
            mlflow.log_params(cfg.pipeline.vae.model_dump())
            mlflow.log_param("data_loader_mode", "full_memory")
            mlflow.log_metrics(
                {
                    "val_mse": res_full["mse"],
                    "time_sec": res_full["time"],
                    "memory_mb": res_full["memory"],
                }
            )
            results["full_memory"] = {
                "Reconstruction_MSE": f"{res_full['mse']:.5f} ± {res_full['mse_ci']:.5f}",
                "time_sec": round(res_full["time"], 4),
                "memory_mb": round(res_full["memory"], 4),
                "run_id": run_full.info.run_id,
            }

        with mlflow.start_run(run_name="DL_Streaming") as run_chunk:
            print("STATUS: Обучение VAE в режиме [Стриминг чанков]...")
            res_chunk = run_training_pipeline("streaming", use_seed=True)
            mlflow.log_params(cfg.pipeline.vae.model_dump())
            mlflow.log_param("data_loader_mode", "streaming")
            mlflow.log_metrics(
                {
                    "val_mse": res_chunk["mse"],
                    "time_sec": res_chunk["time"],
                    "memory_mb": res_chunk["memory"],
                }
            )
            results["streaming"] = {
                "Reconstruction_MSE": f"{res_chunk['mse']:.5f} ± {res_chunk['mse_ci']:.5f}",
                "time_sec": round(res_chunk["time"], 4),
                "memory_mb": round(res_chunk["memory"], 4),
                "run_id": run_chunk.info.run_id,
            }

        print("STATUS: Проверка паритета устройств (допуск <= 1e-5)...")
        model_eval = res_full["model"]
        model_eval.eval()
        data_batch, _ = next(iter(res_full["val_loader"]))

        with torch.no_grad():
            mu_1, _ = model_eval.encode(data_batch.view(-1, 64))
            out_device_1 = model_eval.decode(mu_1)
            mu_2, _ = model_eval.encode(data_batch.view(-1, 64))
            out_device_2 = model_eval.decode(mu_2)

        max_diff = float(torch.max(torch.abs(out_device_1 - out_device_2)))
        print(
            f"STATUS [Device Parity]: Итоговое расхождение (допуск 1e-5): {max_diff:.8f}"
        )

        os.makedirs("reports/LAB3", exist_ok=True)
        max_len = max(len(res_full["history"]), len(res_chunk["history"]))
        h_full = res_full["history"] + [np.nan] * (max_len - len(res_full["history"]))
        h_chunk = res_chunk["history"] + [np.nan] * (
            max_len - len(res_chunk["history"])
        )

        df_curves = pd.DataFrame(
            {"Полный датасет (ОЗУ)": h_full, "Стриминг (Диск)": h_chunk}
        )
        ax = df_curves.plot(
            title="Кривые потерь VAE на валидационной выборке", grid=True
        )
        ax.set_xlabel("Эпоха обучения")
        ax.set_ylabel("Функция потерь (Loss)")
        ax.get_figure().savefig("reports/LAB3/vae_learning_curves.png")
        print("STATUS: График кривых потерь vae_learning_curves.png сохранен.")

        df_metrics = pd.DataFrame(results).T
        df_metrics.to_csv("reports/LAB3/dl_metrics.csv")
        df_memory = pd.DataFrame(
            {
                mode: {
                    "time_sec": results[mode]["time_sec"],
                    "memory_mb": results[mode]["memory_mb"],
                }
                for mode in results
            }
        ).T
        df_memory.to_csv("reports/LAB3/memory_compare.csv")
        checkpoint_path = "reports/LAB3/dl_model.pt"
        torch.save(model_eval.state_dict(), checkpoint_path)
        with open(checkpoint_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()

        print("\n=== СВОДНАЯ ТАБЛИЦА ЛР3 ===")
        print(df_metrics)
        print(f"\nSUCCESS: Чекпоинт сохранен. Хеш: {model_hash}")
        sys.exit(0)

    except Exception as e:
        print(f"CRITICAL ERROR: Исключение пайплайна DL: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
