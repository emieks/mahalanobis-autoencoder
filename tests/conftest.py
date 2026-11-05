import numpy as np
import pandas as pd
import pytest

SENSORS = ["Vibration Radial", "Vibration Axial", "Humidity", "Temperature", "Shaft speed"]
COUPLED = ["Vibration Radial", "Vibration Axial", "Shaft speed"]
FEATURES = ["Time", *SENSORS]


def make_frame(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    speed = rng.choice([0, 300, 1200], size=rows) + rng.normal(0, 5, rows)
    speed = np.maximum(speed, 0)
    radial = 0.03 + 0.0006 * speed + rng.normal(0, 0.01, rows)
    axial = 0.02 + 0.6 * radial + rng.normal(0, 0.01, rows)
    return pd.DataFrame(
        {
            "Time": np.arange(rows) % 86400,
            "Vibration Radial": np.abs(radial),
            "Vibration Axial": np.abs(axial),
            "Humidity": rng.integers(40, 50, rows),
            "Temperature": rng.integers(20, 30, rows),
            "Shaft speed": np.round(speed),
        }
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame(5000)
