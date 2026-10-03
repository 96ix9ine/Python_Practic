from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import os


class DataConfig(BaseModel):
    dataset_name: str = "load_digits"
    n_classes: int = Field(default=10, ge=2, le=10)
    test_size_split: float = Field(default=0.2, ge=0.0, le=0.5)
    random_seed: int = 42
    expected_hash: str = (
        "6e928524d187dc6e37051cde0cc17102b517fd86fae6eb3aab6b83b1bf98e729"
    )


class EDAConfig(BaseModel):
    output_dir: str = "reports/LAB1"
    stats_filename: str = "eda_stats.csv"
    manifest_filename: str = "hash_manifest.json"
    low_variance_threshold: float = 1.0


class Lab1Config(BaseModel):
    data: DataConfig = DataConfig()
    eda: EDAConfig = EDAConfig()


class ModelConfig(BaseModel):
    model_type: str = "logistic_regression"
    pca_components: int = Field(default=16, ge=2, le=64)
    alpha: float = Field(default=0.0001, gt=0)
    random_seed: int = Field(default=42, ge=0)

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed = ["logistic_regression", "knn"]
        if v.lower() not in allowed:
            raise ValueError(f"Недопустимая модель: '{v}'. Разрешены только: {allowed}")
        return v.lower()


class VAEConfig(BaseModel):
    input_dim: int = 64
    latent_dim: int = Field(default=16, ge=2, le=32)  # Сжатие: 64->16->64
    hidden_dim: int = Field(default=32, ge=8)
    lr: float = Field(default=5e-4, gt=0)
    early_stopping_patience: int = Field(default=5, ge=1)


class ValidatorConfig(BaseModel):
    reconstruction_quantile: float = Field(default=0.95, ge=0.5, le=0.99)
    max_false_rejection_rate: float = Field(default=0.05, ge=0.01, le=0.20)
    anomaly_shift_magnitude: float = Field(default=5.0, gt=0.0)


class GeneratorConfig(BaseModel):
    mode: str = "similar"
    num_samples: int = Field(default=500, ge=10, le=10000)
    random_seed: int = Field(default=42, ge=0)

    drift_mean_shift: float = Field(default=0.0, ge=0.0, le=10.0)  # Пиксели
    drift_class_imbalance: bool = Field(
        default=False
    )  # Искажение долей классов (урезание редких)
    drift_flip_labels: bool = Field(default=False)
    drift_contrast_seasonal: float = Field(default=1.0, ge=0.1, le=3.0)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = ["similar", "random"]
        if v.lower() not in allowed:
            raise ValueError(
                f"Недопустимый режим генератора: '{v}'. Разрешены: {allowed}"
            )
        return v.lower()


class PipelineConfig(BaseModel):
    data_path: str = "data/raw/digits.csv"
    chunk_size: int = Field(default=256, ge=16, le=1000)
    batch_size: int = Field(default=128, ge=16)
    epochs: int = Field(default=50, ge=1)
    no_accel: bool = False
    data_loader_mode: str = "full_memory"

    model: ModelConfig = ModelConfig()
    vae: VAEConfig = VAEConfig()
    validator: ValidatorConfig = ValidatorConfig()
    generator: GeneratorConfig = GeneratorConfig()

    @field_validator("data_loader_mode")
    @classmethod
    def validate_loader_mode(cls, v: str) -> str:
        allowed = ["full_memory", "streaming"]
        if v.lower() not in allowed:
            raise ValueError(f"Невалидный режим загрузки: '{v}'")
        return v.lower()

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v: str) -> str:
        if not os.path.exists(v) and v == "data/raw/digits.csv":
            pass
        if not v.endswith(".csv"):
            raise ValueError("Файл данных должен быть формата .csv")
        return v


class Lab2Config(BaseModel):
    pipeline: PipelineConfig = PipelineConfig()


class Lab3Config(BaseModel):
    pipeline: PipelineConfig = PipelineConfig()


class Lab4Config(BaseModel):
    pipeline: PipelineConfig = PipelineConfig()


class Lab5Config(BaseModel):
    pipeline: PipelineConfig = PipelineConfig()
