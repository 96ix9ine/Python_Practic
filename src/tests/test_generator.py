import pytest
import numpy as np
import pandas as pd
from configs.config_schema import Lab5Config
from src.data.synthetic_generator import ControlledSyntheticGenerator


@pytest.fixture
def base_config():
    return Lab5Config()


def test_reproducibility(base_config):
    """Проверка воспроизводимости по зерну (Негативный контроль)"""
    base_config.pipeline.generator.random_seed = 42
    gen1 = ControlledSyntheticGenerator(base_config)
    df1 = gen1.generate_dataset()

    gen2 = ControlledSyntheticGenerator(base_config)
    df2 = gen2.generate_dataset()

    assert np.allclose(df1.values, df2.values)


def test_drift_mean_shift(base_config):
    """Тест ручки дрейфа сдвига средних яркостей пикселей"""
    base_config.pipeline.generator.drift_mean_shift = 0.0
    df_clean = ControlledSyntheticGenerator(base_config).generate_dataset()

    base_config.pipeline.generator.drift_mean_shift = 3.0
    df_drift = ControlledSyntheticGenerator(base_config).generate_dataset()

    assert np.mean(df_drift.drop(columns=["target"]).values) > np.mean(
        df_clean.drop(columns=["target"]).values
    )


def test_drift_class_imbalance(base_config):
    """Тест ручки дрейфа дисбаланса классов"""
    base_config.pipeline.generator.drift_class_imbalance = True
    df = ControlledSyntheticGenerator(base_config).generate_dataset()

    assert int(np.sum(df["target"] > 5)) == 0


def test_drift_flip_labels(base_config):
    """Тест ручки дрейфа инверсии (переворота меток/цветов)"""
    base_config.pipeline.generator.drift_flip_labels = True
    df = ControlledSyntheticGenerator(base_config).generate_dataset()

    assert np.max(df.drop(columns=["target"]).values) <= 16.0
    assert np.min(df.drop(columns=["target"]).values) >= 0.0
