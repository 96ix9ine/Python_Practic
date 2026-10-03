import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from configs.config_schema import Lab2Config


class SklearnIncrementalPipeline:
    """ML-конвейер на базе стандартного sklearn.pipeline.Pipeline"""

    def __init__(self, cfg: Lab2Config):
        self.cfg = cfg.pipeline.model

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", IncrementalPCA(n_components=self.cfg.pca_components)),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=self.cfg.alpha,
                        random_state=self.cfg.random_seed,
                    ),
                ),
            ]
        )
        self.classes_ = np.arange(10)

    def fit_full(self, X: np.ndarray, y: np.ndarray) -> None:
        """Полноразмерное обучение пайплайна целиком"""
        self.pipeline.fit(X, y)

    def partial_fit_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> None:
        """Инкрементальное обучение по чанкам"""
        scaler = self.pipeline.named_steps["scaler"]
        pca = self.pipeline.named_steps["pca"]
        classifier = self.pipeline.named_steps["classifier"]

        scaler.partial_fit(X_chunk)
        X_scaled = scaler.transform(X_chunk)

        if len(X_chunk) >= pca.n_components:
            pca.partial_fit(X_scaled)
            X_pca = pca.transform(X_scaled)
            classifier.partial_fit(X_pca, y_chunk, classes=self.classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание через стандартный интерфейс Pipeline"""
        return self.pipeline.predict(X)
