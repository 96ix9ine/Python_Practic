import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitsVAE(nn.Module):
    """VAE Архитектура для Варианта 16 (64 -> 16 -> 64)"""

    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=16):
        super(DigitsVAE, self).__init__()

        # Энкодер
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # Декодер
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """Трюк репараметризации"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 64))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    """Loss = Ошибка реконструкции (MSE) + Расхождение Кульбака-Лейблера (KLD)"""
    MSE = F.mse_loss(recon_x, x.view(-1, 64), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD
