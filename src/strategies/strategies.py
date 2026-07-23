import pandas as pd


def get_ema_crossover_positions(
    fast_ema: pd.Series,
    slow_ema: pd.Series,
    long_only: bool = True) -> pd.Series:
    """
    Return target positions from a basic EMA crossover rule.
    """
    fast_ema, slow_ema = fast_ema.align(slow_ema, join="inner")

    bullish = fast_ema > slow_ema

    if long_only:
        positions = bullish.astype(float)
    else:
        positions = bullish.astype(float).replace({0.0: -1.0})

    positions.name = "target_position"
    return positions