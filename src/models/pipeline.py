import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from configs.config_schema import Lab2Config


class DigitsPipeline:
    """Конвейер: Масштабирование + PCA + Логистическая регрессия

    Поддерживает как полное обучение (fit), так и инкрементальное по чанкам
    (partial_fit).
    """

    def __init__(self, cfg: Lab2Config):
        self.cfg = cfg.pipeline.model

        # Инициализируем компоненты с поддержкой частичного обучения
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=self.cfg.pca_components)
        self.model = SGDClassifier(
            loss="log_loss",
            alpha=self.cfg.alpha,
            random_state=self.cfg.random_seed,
        )
        self.classes_ = np.arange(10)

    def fit_full(self, X: np.ndarray, y: np.ndarray):
        """Обучение на полном датасете"""
        print("STATUS [Pipeline]: Запуск полноразмерного обучения конвейера...")
        try:
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)
            self.model.fit(X_pca, y)
        except Exception as e:
            print(f"CRITICAL ERROR [Pipeline]: Сбой при полном обучении: {str(e)}")
            sys.exit(1)

    def partial_fit_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray):
        """Инкрементальное обучение на одном чанке"""
        try:
            # Частичное накопление статистик масштабирования
            self.scaler.partial_fit(X_chunk)
            X_scaled = self.scaler.transform(X_chunk)

            # Частичное обучение PCA (требует, чтобы размер чанка был > n_components)
            if len(X_chunk) >= self.pca.n_components:
                self.pca.partial_fit(X_scaled)
                X_pca = self.pca.transform(X_scaled)
                # Инкрементальное обучение логистической регрессии
                self.model.partial_fit(X_pca, y_chunk, classes=self.classes_)
            else:
                pass
        except Exception as e:
            print(f"CRITICAL ERROR [Pipeline]: Сбой при обучении чанка: {str(e)}")
            sys.exit(1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Применение конвейера для предсказания"""
        try:
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
            return self.model.predict(X_pca)
        except Exception as e:
            print(f"CRITICAL ERROR [Pipeline]: Сбой при предсказании: {str(e)}")
            sys.exit(1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Вычисление базовой метрики Accuracy"""
        from sklearn.metrics import accuracy_score

        preds = self.predict(X)
        return float(accuracy_score(y, preds))
