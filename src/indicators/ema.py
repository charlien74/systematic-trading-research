import pandas as pd

def get_ema_weights(period: int, halflife: int) -> pd.Series:
    """
    Return exponential moving average weights for a given period.
    Weights are newest -> oldest (exponentially decreasing).
    """
    base = 2 ** (1 / halflife)
    weights = base ** pd.Series(range(period))
    weights /= weights.sum()
    return weights

def get_ema_series(data: pd.DataFrame, 
                   period: int, 
                   halflife: int, 
                   column: str = "Close") -> pd.Series:
    """
    Get the exponential moving average series for a given period and halflife.
    """
    weights = get_ema_weights(period, halflife)
    ema_series = data[column].rolling(window=period).apply(lambda x: (x * weights).sum(), raw=True)
    return ema_series