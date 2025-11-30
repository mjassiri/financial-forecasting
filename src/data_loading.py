from pathlib import Path
import pandas as pd

# Base data directory: .../financial-forecasting/data
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw_series(filename: str, value_col: str = "Close") -> pd.Series:
    """
    Load a raw CSV from data/raw and return a cleaned time series.

    - filename: name of the CSV file in data/raw
    - value_col: which column to use as the series values (e.g. 'Close')
    """
    path = DATA_DIR / "raw" / filename

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    df = df.set_index("Date")

    # Take the value column as a float Series
    series = df[value_col].astype(float)
    series.name = value_col

    # Drop any missing values just in case
    series = series.dropna()

    return series


def train_val_split(series: pd.Series, val_fraction: float = 0.2):
    """
    Split a series into train and validation sets by time.
    """
    n = len(series)
    split_idx = int(n * (1 - val_fraction))
    train = series.iloc[:split_idx]
    val = series.iloc[split_idx:]
    return train, val
