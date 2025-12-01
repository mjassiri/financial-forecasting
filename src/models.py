import pandas as pd


def naive_forecast(train: pd.Series, val: pd.Series) -> pd.Series:
    """
    Naive forecast: each day's prediction is just the previous day's actual value.
    We build it over the combined series so that validation predictions
    always use only past information.
    """
    full = pd.concat([train, val])
    shifted = full.shift(1)     # yesterday's value
    preds = shifted.loc[val.index]
    preds.name = "naive"
    return preds


def moving_average_forecast(train: pd.Series, val: pd.Series, window: int = 30) -> pd.Series:
    """
    Moving-average forecast: each prediction is the rolling mean of the last `window` days.
    """
    full = pd.concat([train, val])
    rolling = full.rolling(window=window).mean()
    preds = rolling.loc[val.index]
    preds.name = f"ma_{window}"
    return preds
