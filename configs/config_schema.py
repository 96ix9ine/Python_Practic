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
    # kNN/LogReg ± PCA
    model_type: str = "logistic_regression"
    pca_components: int = Field(default=16, ge=2, le=64)
    alpha: float = Field(
        default=0.0001, gt=0
    )  # Коэффициент регуляризации для SGD/LogReg
    random_seed: int = Field(default=42, ge=0)

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed = ["logistic_regression", "knn"]
        if v.lower() not in allowed:
            raise ValueError(f"Недопустимая модель: '{v}'. Разрешены только: {allowed}")
        return v.lower()


class PipelineConfig(BaseModel):
    data_path: str = "data/raw/digits.csv"
    chunk_size: int = Field(
        default=256, ge=16, le=1000
    )  # Размер чанка для режима потока
    batch_size: int = Field(default=128, ge=16)  # Задел под VAE из примера
    epochs: int = Field(default=10, ge=1)
    no_accel: bool = False

    model: ModelConfig = ModelConfig()

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v: str) -> str:
        if not os.path.exists(v) and v == "data/raw/digits.csv":
            pass
        if not v.endswith(".csv"):
            raise ValueError(f"Файл данных должен быть формата .csv, получено: '{v}'")
        return v


class Lab2Config(BaseModel):
    pipeline: PipelineConfig = PipelineConfig()
