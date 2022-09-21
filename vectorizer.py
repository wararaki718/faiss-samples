import numpy as np
import pandas as pd


def min_max_scaler(x: np.ndarray) -> np.ndarray:
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def vectorize(df: pd.DataFrame) -> np.ndarray:
    features = pd.concat([
        df,
        pd.get_dummies(df["Type 1"]),
        pd.get_dummies(df["Type 2"])
    ], axis=1)
    features.drop(["Name", "Type 1", "Type 2"], axis=1, inplace=True)
    features["Legendary"] = features["Legendary"].astype(int)
    features = features.values.astype(np.float32)
    features = min_max_scaler(features)
    features = np.ascontiguousarray(features, dtype=np.float32)
    return features
