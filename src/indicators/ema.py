import pandas as pd


def get_ema_series(data: pd.DataFrame, 
                   period: int, 
                   halflife: int, 
                   column: str = "Close") -> pd.Series:
    """
    Get the exponential moving average series for a given period and halflife.
    """
    return data[column].ewm(halflife=halflife, adjust=False).mean().tail(period)
