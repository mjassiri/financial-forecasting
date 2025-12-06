from statsmodels.tsa.arima.model import ARIMA
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

def arima_forecast(train: pd.Series, val: pd.Series, order=(5, 1, 0)) -> pd.Series:
    """
    ARIMA forecasting model.
    order = (p, d, q) controls the autoregressive, differencing, and moving average parts.
    """
    # Fit the model on the training data only
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    # Forecast the next len(val) points
    preds = model_fit.forecast(steps=len(val))

    # Align index with validation dates
    preds.index = val.index
    preds.name = f"arima_{order}"
    return preds

