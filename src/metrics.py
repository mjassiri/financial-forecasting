import numpy as np
import pandas as pd


def _to_numpy(y_true, y_pred):
    """Helper to convert inputs to numpy arrays."""
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = _to_numpy(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = _to_numpy(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (in %)."""
    y_true, y_pred = _to_numpy(y_true, y_pred)
    eps = 1e-8  # avoid divide-by-zero
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)
