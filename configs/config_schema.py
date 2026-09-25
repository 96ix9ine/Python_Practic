from pydantic import BaseModel, Field
from typing import List, Optional


class DataConfig(BaseModel):
    dataset_name: str = "load_digits"
    n_classes: int = Field(default=10, ge=2, le=10)
    test_size_split: float = Field(default=0.2, ge=0.0, le=0.5)
    random_seed: int = 42


class EDAConfig(BaseModel):
    output_dir: str = "reports/LAB_1"
    stats_filename: str = "eda_stats.csv"
    manifest_filename: str = "hash_manifest.json"
    low_variance_threshold: float = 1.0


class Lab1Config(BaseModel):
    data: DataConfig = DataConfig()
    eda: EDAConfig = EDAConfig()
