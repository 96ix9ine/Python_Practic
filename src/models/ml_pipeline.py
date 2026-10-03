import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from configs.config_schema import Lab2Config


class SklearnIncrementalPipeline:
    """Инкрементальный конвейер sklearn.pipeline"""

    def __init__(self, cfg: Lab2Config):
        self.cfg = cfg.pipeline.model

        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=self.cfg.pca_components)
        self.classifier = SGDClassifier(
            loss="log_loss", alpha=self.cfg.alpha, random_state=self.cfg.random_seed
        )
        self.classes_ = np.arange(10)

    def fit_full(self, X: np.ndarray, y: np.ndarray) -> None:
        """Полноразмерное обучение на всем датасете"""
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.classifier.fit(X_pca, y)

    def partial_fit_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> None:
        """Последовательное обучение чанками без загрузки всего датасета"""
        self.scaler.partial_fit(X_chunk)
        X_scaled = self.scaler.transform(X_chunk)

        if len(X_chunk) >= self.pca.n_components:
            self.pca.partial_fit(X_scaled)
            X_pca = self.pca.transform(X_scaled)
            self.classifier.partial_fit(X_pca, y_chunk, classes=self.classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.classifier.predict(X_pca)
