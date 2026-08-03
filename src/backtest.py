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

def compute_strategy_returns_asset_pair(
    y_asset_returns: pd.Series, 
    x_asset_returns: pd.Series,
    positions: pd.DataFrame,
    method: str = "simple",
) -> pd.Series:
    """
    Compute strategy returns for a strategy involving 2 assets x and y
    Positions should have cols 'x_position' and 'y_position'.
    """
    required_columns = {"x_position", "y_position"}
    missing_columns = required_columns - set(positions.columns)
    if missing_columns:
        raise ValueError(
            "positions DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    x_positions = positions["x_position"]
    y_positions = positions["y_position"]
    
    x_asset_returns, x_positions = x_asset_returns.align(
        x_positions,
        join="inner",
    )
    y_asset_returns, y_positions = y_asset_returns.align(
        y_positions,
        join="inner",
    )

    _validate_return_method(method)

    strategy_returns = x_positions.shift(1) * x_asset_returns + y_positions.shift(1) * y_asset_returns
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
        transaction_cost_bps: float = 0.0
) -> dict:
    """
    Wrapper function to perform all backtest computations from positions and price data.
    """
    asset_returns = compute_simple_asset_returns(prices)
    strategy_returns = compute_strategy_returns(asset_returns, positions)
    # Apply transaction costs if specified
    if transaction_cost_bps > 0.0:
        returns_with_costs = apply_transaction_costs(strategy_returns, 
                                                     positions, 
                                                     cost_bps=transaction_cost_bps)
        strategy_returns = returns_with_costs["net_return"]
    exposed_fraction = positions.abs().mean()
    n_trades = positions.diff().abs().sum()
    benchmark_returns = compute_benchmark_returns(asset_returns, tail=len(strategy_returns))
    strategy_cumulative_returns = aggregate_returns(strategy_returns, method="simple", cumulative=True)
    benchmark_cumulative_returns = aggregate_returns(benchmark_returns, method="simple", cumulative=True)
    metrics = compute_metrics(strategy_returns, benchmark_returns, periods_per_year=periods_per_year)
    metrics["exposed_fraction"] = exposed_fraction
    metrics["number_trades"] = n_trades
    return metrics, strategy_cumulative_returns, benchmark_cumulative_returns

def permute_prices(prices: pd.Series | pd.DataFrame, 
                   column: str = "Close",
                   seed: int | None = None) -> pd.Series:
    """
    Randomly permute returns of a price series, preserving the original index.
    """
    if isinstance(prices, pd.DataFrame):
        if column not in prices.columns:
            raise KeyError(f"Column '{column}' not found in prices DataFrame")
        price_series = prices[column]
    else:
        price_series = prices

    seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)

    simple_returns = price_series.pct_change().dropna()
    permuted_returns = simple_returns.reindex(np.random.permutation(simple_returns.index))

    start_price = float(price_series.iloc[0])
    growth_path = (1.0 + permuted_returns).cumprod()
    permuted_prices = pd.concat(
        [pd.Series([start_price]), start_price * growth_path],
        ignore_index=True,
    )
    permuted_prices.index = price_series.index
    permuted_prices.name = price_series.name

    return permuted_prices


def apply_transaction_costs(
    gross_returns: pd.Series,
    target_positions: pd.DataFrame | pd.Series,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Deduct proportional transaction costs based on portfolio turnover.

    Parameters
    ----------
    gross_returns:
        Strategy returns before transaction costs.

    target_positions:
        Target asset weights. If a dataframe, expected columns include the 
        positions used to calculate strategy returns. If a series, it is assumed 
        to be the position for a single asset.

    cost_bps:
        One-way transaction cost in basis points per unit of notional
        traded. For example, 5 means 0.05%.

    Returns
    -------
    pd.DataFrame
        Gross returns, turnover, transaction costs, and net returns.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be nonnegative.")

    cost_rate = cost_bps / 10_000.0

    if isinstance(target_positions, pd.Series):
        weights = target_positions.to_frame().fillna(0.0)
    else:
        if {"x_position", "y_position"}.issubset(target_positions.columns):
            # Keep backward-compatible behavior for pair strategies.
            weights = target_positions[["x_position", "y_position"]].fillna(0.0)
        else:
            numeric_columns = target_positions.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                raise ValueError("target_positions DataFrame must contain numeric position columns.")
            weights = target_positions[numeric_columns].fillna(0.0)

    turnover = weights.diff().abs().sum(axis=1)

    # Treat the first target allocation as trading from a flat portfolio.
    turnover.iloc[0] = weights.iloc[0].abs().sum()

    # Returns at time t are generated by positions at time t-1, so costs must be
    # shifted to the same return period.
    turnover_for_returns = turnover.shift(1)
    transaction_cost = cost_rate * turnover_for_returns

    gross_returns, transaction_cost = gross_returns.align(
        transaction_cost,
        join="inner",
    )

    return pd.DataFrame(
        {
            "gross_return": gross_returns,
            "turnover": turnover_for_returns.reindex(gross_returns.index),
            "transaction_cost": transaction_cost,
            "net_return": gross_returns - transaction_cost,
        }
    )
