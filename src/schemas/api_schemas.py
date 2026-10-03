from pydantic import BaseModel, Field, field_validator
from typing import List


class PredictRequest(BaseModel):
    pixels: List[float] = Field(
        ..., description="Массив из 64 значений яркости пикселей (0-16)"
    )

    @field_validator("pixels")
    @classmethod
    def validate_pixels_dim_and_range(cls, v: List[float]) -> List[float]:
        if len(v) != 64:
            raise ValueError(
                f"Размерность вектора пикселей должна быть строго 64, получено {len(v)}"
            )
        for idx, val in enumerate(v):
            if val < 0.0 or val > 16.0:
                raise ValueError(
                    f"Значение пикселя под индексом {idx} вышло за пределы: {val}"
                )
        return v


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Предсказанный класс цифры (0-9)")
    is_anomaly: bool = Field(
        ..., description="Флаг аномалии (Out-of-Distribution detector контур)"
    )
    model_version: str = Field(..., description="Версия модели, обработавшая запрос")


class VersionInfo(BaseModel):
    version: str
    path: str
    hash: str
    status: str  # "готовится", "пробная", "рабочая", "снята"
    accuracy: float


class RegistrySchema(BaseModel):
    versions: List[VersionInfo]
    active_version: str
