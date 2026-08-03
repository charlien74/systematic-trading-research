import numpy as np
import pandas as pd
import pytest

from src.backtest import apply_transaction_costs, permute_prices

def test_permute_prices_preserves_border_values() -> None:
    # Note: this fails if price is ever zero, as returns become infinite.
    prices = pd.Series(np.arange(1, 11, dtype=float), name="price")
    permuted_prices = permute_prices(prices)

    assert permuted_prices.iloc[0] == prices.iloc[0]
    assert permuted_prices.iloc[-1] == prices.iloc[-1]
    # Make sure prices remains the same
    assert all(np.isclose(prices.iloc[i], np.float64(i + 1)) for i in range(10))


def test_apply_transaction_costs_single_asset_timing_alignment() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    gross_returns = pd.Series([np.nan, 0.10, 0.00, 0.00], index=idx)
    positions = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)

    out = apply_transaction_costs(gross_returns, positions, cost_bps=10.0)

    assert np.isnan(out.loc[idx[0], "transaction_cost"])
    assert out.loc[idx[1], "transaction_cost"] == pytest.approx(0.001)
    assert out.loc[idx[2], "transaction_cost"] == pytest.approx(0.0)
    assert out.loc[idx[3], "transaction_cost"] == pytest.approx(0.001)
    assert out.loc[idx[1], "net_return"] == pytest.approx(0.099)
    assert out.loc[idx[3], "net_return"] == pytest.approx(-0.001)


def test_apply_transaction_costs_pair_uses_x_y_columns_when_present() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    gross_returns = pd.Series([np.nan, 0.00, 0.00, 0.00], index=idx)
    positions = pd.DataFrame(
        {
            "x_position": [0.5, 0.5, 0.5, 0.5],
            "y_position": [-0.5, -0.5, -0.5, -0.5],
            # This column should not affect turnover when x/y are present.
            "spread_position": [1.0, -1.0, 1.0, -1.0],
        },
        index=idx,
    )

    out = apply_transaction_costs(gross_returns, positions, cost_bps=10.0)

    assert out.loc[idx[1], "transaction_cost"] == pytest.approx(0.001)
    assert out.loc[idx[2], "transaction_cost"] == pytest.approx(0.0)
    assert out.loc[idx[3], "transaction_cost"] == pytest.approx(0.0)


def test_apply_transaction_costs_rejects_negative_cost_bps() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    gross_returns = pd.Series([np.nan, 0.01], index=idx)
    positions = pd.Series([1.0, 1.0], index=idx)

    with pytest.raises(ValueError, match="nonnegative"):
        apply_transaction_costs(gross_returns, positions, cost_bps=-1.0)