import os
import numpy as np
import pandas as pd
from configs.config_schema import Lab5Config


class ControlledSyntheticGenerator:
    """Управляемый параметрический генератор синтетических изображений цифр по профилю P2"""

    def __init__(self, cfg: Lab5Config, real_data_path: str = "data/raw/digits.csv"):
        self.cfg = cfg.pipeline.generator
        self.real_data_path = real_data_path

        if os.path.exists(real_data_path):
            df_real = pd.read_csv(real_data_path)
            self.X_real = df_real.drop(columns=["target"]).values
            self.y_real = df_real["target"].values
        else:
            self.X_real = np.zeros((100, 64))
            self.y_real = np.zeros(100)

    def generate_dataset(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.cfg.random_seed)
        n_samples = self.cfg.num_samples

        if self.cfg.mode == "similar":
            indices = rng.choice(len(self.X_real), size=n_samples, replace=True)
            X_synth = self.X_real[indices].copy().astype(float)
            y_synth = self.y_real[indices].copy()

            X_synth += rng.normal(loc=0.0, scale=0.1, size=X_synth.shape)
        else:
            X_synth = rng.uniform(0.0, 16.0, size=(n_samples, 64))
            y_synth = rng.choice(np.unique(self.y_real), size=n_samples)

        if self.cfg.drift_mean_shift > 0.0:
            X_synth += self.cfg.drift_mean_shift

        if self.cfg.drift_contrast_seasonal != 1.0:
            X_synth = (
                X_synth - np.mean(X_synth, axis=0)
            ) * self.cfg.drift_contrast_seasonal + np.mean(X_synth, axis=0)

        X_synth = np.clip(np.round(X_synth), 0.0, 16.0)

        if self.cfg.drift_class_imbalance:
            y_synth[y_synth > 5] = 0

        if self.cfg.drift_flip_labels:
            X_synth = 16.0 - X_synth

        columns = [f"pixel_{i}" for i in range(64)]
        df_out = pd.DataFrame(X_synth, columns=columns)
        df_out["target"] = y_synth.astype(int)
        return df_out
