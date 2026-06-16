#!/usr/bin/env python3
# Data loading and preprocessing utilities

import numpy as np
import pandas as pd
from typing import Tuple

FEATURES = ['latitude', 'longitude', 'laser_alt', 'magnetics_final_microlevelled']


def load_magnetic_csv(path: str) -> pd.DataFrame:
    """Load magnetic CSV into a DataFrame."""
    return pd.read_csv(path)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].values


def train_test_split_scaled(data: np.ndarray, test_size: float = 0.2, random_state: int = 42, mag_weight: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test and scale using train stats. Returns X_train, X_test, X_full_scaled."""
    from sklearn.model_selection import train_test_split

    X_train_raw, X_test_raw = train_test_split(data, test_size=test_size, random_state=random_state)

    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    X_full_scaled = (data - mean) / std

    # Apply magnetic weight to the fourth column
    X_train[:, 3] *= mag_weight
    X_test[:, 3] *= mag_weight
    X_full_scaled[:, 3] *= mag_weight

    return X_train, X_test, X_full_scaled
