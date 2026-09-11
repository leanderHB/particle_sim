import numpy as np


def first_hot_then_cold(t: float) -> float:
    down_point = 1500
    up_point = 100
    width = 20
    return (
        5
        / (1 + np.exp(1 * (t - down_point) / width))
        / (1 + np.exp(-1 * (t - up_point) / width))
        + 0.5
    )


TEMP_CURVES = {
    "first_hot_then_cold": first_hot_then_cold,
}
