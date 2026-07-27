import numpy as np
import pandas as pd
from src.backtest import permute_prices

def test_permute_prices_preserves_border_values() -> None:
    prices = pd.Series(np.arange(10, dtype=float), name="price")
    permuted_prices = permute_prices(prices)

    assert permuted_prices.iloc[0] == prices.iloc[0]
    assert permuted_prices.iloc[-1] == prices.iloc[-1]
    # Make sure prices remains the same
    assert all(prices.iloc[i] == float(i) for i in range(10))