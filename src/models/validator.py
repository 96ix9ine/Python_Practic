import numpy as np
import torch
import os
from configs.config_schema import Lab4Config
from src.models.vae_model import DigitsVAE


class ImageOODValidator:
    """Модель-валидатор (OOD) на основе ошибки реконструкции VAE"""

    def __init__(
        self, cfg: Lab4Config, checkpoint_path: str = "reports/LAB3/dl_model.pt"
    ):
        self.cfg = cfg.pipeline.validator
        self.device = torch.device("cpu")

        self.vae = DigitsVAE()
        if os.path.exists(checkpoint_path):
            self.vae.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
        self.vae.eval()

        self.mean_mse = 0.0
        self.threshold_mse = 0.0

    def _calculate_sample_mses(self, X: np.ndarray) -> np.ndarray:
        """Внутренний расчет MSE реконструкции пикселей средствами PyTorch/NumPy"""
        X_tensor = torch.tensor(X, dtype=torch.float32) / 16.0
        with torch.no_grad():
            mu, _ = self.vae.encode(X_tensor)
            recon = self.vae.decode(mu)
            mse = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
        return mse

    def fit_full(self, X: np.ndarray) -> None:
        """Обучение статистик валидатора на полном датасете"""
        mses = self._calculate_sample_mses(X)
        self.mean_mse = float(np.mean(mses))
        self.threshold_mse = float(np.quantile(mses, self.cfg.reconstruction_quantile))

    def partial_fit_chunk(self, X_chunk: np.ndarray, chunk_index: int) -> None:
        """Инкрементальное накопление статистик средних по чанкам"""
        chunk_mses = self._calculate_sample_mses(X_chunk)
        chunk_mean = np.mean(chunk_mses)

        if chunk_index == 0:
            self.mean_mse = float(chunk_mean)
            self.threshold_mse = float(
                np.quantile(chunk_mses, self.cfg.reconstruction_quantile)
            )
        else:
            self.mean_mse = float(0.8 * self.mean_mse + 0.2 * chunk_mean)
            self.threshold_mse = float(
                0.8 * self.threshold_mse
                + 0.2 * np.quantile(chunk_mses, self.cfg.reconstruction_quantile)
            )

    def predict_rejection(self, X: np.ndarray) -> np.ndarray:
        """Возвращает 1 (ОТКАЗ/АНОМАЛИЯ), если MSE выше порога, иначе 0"""
        mses = self._calculate_sample_mses(X)
        return (mses > self.threshold_mse).astype(int)

    def save_params(self, path: str) -> None:
        """Сохранение параметров статистик в формате numpy npz"""
        np.savez(path, mean_mse=self.mean_mse, threshold_mse=self.threshold_mse)

    def load_params(self, path: str) -> None:
        data = np.load(path)
        self.mean_mse = float(data["mean_mse"])
        self.threshold_mse = float(data["threshold_mse"])
