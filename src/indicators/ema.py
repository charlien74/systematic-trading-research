import pandas as pd


def get_ema_series(data: pd.DataFrame | pd.Series, 
                   period: int, 
                   halflife: int, 
                   column: str = "Close") -> pd.Series:
    """
    Get the exponential moving average series for a given period and halflife.
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in data DataFrame")
        series = data[column]
    else:
        series = data
    return series.ewm(halflife=halflife, adjust=False).mean().tail(period)
