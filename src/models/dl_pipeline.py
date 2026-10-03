import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.vae_model import vae_loss_function


class VAETrainer:
    """Изолированный DL-пайплайн управления обучением вариационного автоэнкодера"""

    def __init__(self, model: nn.Module, lr: float, device: torch.device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Обучение нейросети в рамках одной эпохи"""
        self.model.train()
        total_loss = 0.0
        samples_count = 0
        for data, _ in dataloader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(data)
            loss = vae_loss_function(recon, data, mu, logvar)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            samples_count += data.size(0)
        return total_loss / samples_count

    def evaluate(self, dataloader: DataLoader) -> float:
        """Валидация/оценка нейросети на проверочной выборке"""
        self.model.eval()
        total_loss = 0.0
        samples_count = 0
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                recon, mu, logvar = self.model(data)
                loss = vae_loss_function(recon, data, mu, logvar)
                total_loss += loss.item()
                samples_count += data.size(0)
        return total_loss / samples_count
