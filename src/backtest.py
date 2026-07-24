import numpy as np
import pandas as pd


def compute_simple_asset_returns(
    prices: pd.Series | pd.DataFrame,
    column: str = "Close",
) -> pd.Series:
    """
    Compute per-period asset returns from a price series.
    """
    if isinstance(prices, pd.DataFrame):
        if column not in prices.columns:
            raise KeyError(f"Column '{column}' not found in prices DataFrame")
        price_series = prices[column]
    else:
        price_series = prices

    price_series = price_series.astype(float)
    returns = price_series.pct_change()

    returns = returns.dropna().astype(float)
    returns.name = "asset_return"
    return returns


def _validate_return_method(method: str) -> None:
    if method not in {"simple", "log"}:
        raise ValueError("method must be either 'simple' or 'log'")


def convert_simple_to_log_returns(simple_returns: pd.Series) -> pd.Series:
    """
    Convert simple returns to log returns.
    """
    valid = simple_returns.dropna()
    if (1.0 + valid <= 0.0).any():
        raise ValueError("Cannot convert to log returns when any simple return <= -1")

    log_returns = np.log1p(simple_returns)
    log_returns.name = simple_returns.name
    return log_returns


def compute_strategy_returns(
    asset_returns: pd.Series,
    positions: pd.Series,
    method: str = "simple",
) -> pd.Series:
    """Compute strategy returns from end-of-period target positions.

    Positions determined at time t are applied to returns at time t+1.
    """
    asset_returns, positions = asset_returns.align(
        positions,
        join="inner",
    )

    _validate_return_method(method)

    strategy_returns = positions.shift(1) * asset_returns
    strategy_returns.name = "strategy_return"

    if method == "log":
        strategy_returns = convert_simple_to_log_returns(strategy_returns)

    return strategy_returns


def compute_benchmark_returns(
    asset_returns: pd.Series,
    tail: int | None = None,
    method: str = "simple",
) -> pd.Series:
    """
    Compute benchmark returns from end-of-period target positions.
    """
    _validate_return_method(method)

    benchmark_returns = asset_returns.copy()
    benchmark_returns.name = "benchmark_return"

    if method == "log":
        benchmark_returns = convert_simple_to_log_returns(benchmark_returns)

    if tail is not None:
        benchmark_returns = benchmark_returns.tail(tail)
    return benchmark_returns

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute the annualized volatility of a return series.
    """
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0

    return float(clean.std() * np.sqrt(periods_per_year))

def aggregate_returns(
    returns: pd.Series,
    method: str = "simple",
    cumulative: bool = False,
) -> float | pd.Series:
    """
    Aggregate per-period returns.

    Args:
        returns: Period return series.
        method: Interpretation of input returns ("simple" or "log").
        cumulative: If True, return cumulative return series over time.
            If False, return final total return scalar.
    """
    _validate_return_method(method)

    clean = returns.dropna().astype(float)
    if clean.empty:
        if cumulative:
            return pd.Series(dtype=float, name=returns.name)
        return 0.0

    if method == "simple":
        if cumulative:
            cumulative_returns = (1.0 + clean).cumprod() - 1.0
            cumulative_returns.name = returns.name
            return cumulative_returns
        return float((1.0 + clean).prod() - 1.0)

    if cumulative:
        cumulative_returns = np.expm1(clean.cumsum())
        cumulative_returns.name = returns.name
        return cumulative_returns

    return float(np.expm1(clean.sum()))

def compute_metrics(
        strategy_simple_returns: pd.Series,
        benchmark_simple_returns: pd.Series,
        periods_per_year: int = 252,
) -> dict:
    """
    Compute performance metrics for a strategy and benchmark.
    """
    strategy_annualized_vol = annualized_volatility(strategy_simple_returns, periods_per_year)
    benchmark_annualized_vol = annualized_volatility(benchmark_simple_returns, periods_per_year)

    strategy_total_return = aggregate_returns(strategy_simple_returns, method="simple")
    benchmark_total_return = aggregate_returns(benchmark_simple_returns, method="simple")

    strategy_sharpe = strategy_simple_returns.mean() / strategy_simple_returns.std() * np.sqrt(periods_per_year) if strategy_simple_returns.std() != 0 else 0.0
    benchmark_sharpe = benchmark_simple_returns.mean() / benchmark_simple_returns.std() * np.sqrt(periods_per_year) if benchmark_simple_returns.std() != 0 else 0.0

    return {
        "strategy_annualized_volatility": strategy_annualized_vol,
        "benchmark_annualized_volatility": benchmark_annualized_vol,
        "strategy_total_return": strategy_total_return,
        "benchmark_total_return": benchmark_total_return,
        "strategy_sharpe": strategy_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
    }

def compute_metrics_and_returns_from_positions_and_prices(
        positions: pd.Series,
        prices: pd.Series | pd.DataFrame,
        periods_per_year: int = 252,
) -> dict:
    """
    Wrapper function to perform all backtest computations from positions and price data.
    """
    asset_returns = compute_simple_asset_returns(prices)
    strategy_returns = compute_strategy_returns(asset_returns, positions)
    benchmark_returns = compute_benchmark_returns(asset_returns, tail=len(strategy_returns))
    strategy_cumulative_returns = aggregate_returns(strategy_returns, method="simple", cumulative=True)
    benchmark_cumulative_returns = aggregate_returns(benchmark_returns, method="simple", cumulative=True)
    return compute_metrics(strategy_returns, benchmark_returns, periods_per_year=periods_per_year), strategy_cumulative_returns, benchmark_cumulative_returns
