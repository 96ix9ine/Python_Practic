import numpy as np
from scipy.stats import ks_2samp


def calculate_psi(
    actual: np.ndarray, expected: np.ndarray, num_buckets: int = 10
) -> float:
    """Расчет Population Stability Index (PSI) для одного непрерывного признака."""
    eps = 1e-4

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5

    expected_counts, _ = np.histogram(expected, bins=buckets)
    actual_counts, _ = np.histogram(actual, bins=buckets)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    # Формула PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def calculate_ks_statistic(actual: np.ndarray, expected: np.ndarray) -> float:
    """Расчет статистики критерия Колмогорова-Смирнова (KS-test)."""
    if (
        np.all(actual == actual[0])
        and np.all(expected == expected[0])
        and actual[0] == expected[0]
    ):
        return 0.0

    statistic, _ = ks_2samp(actual, expected)
    return float(statistic)
