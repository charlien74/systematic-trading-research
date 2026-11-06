"""Chronological, rolling-window selection and evaluation of EMA strategies."""

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from typing import Iterable

import numpy as np
import pandas as pd

from src.backtest import compute_metrics


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    daily_returns: pd.DataFrame
    training_scores: pd.DataFrame


def _returns_for_positions(
    asset_returns: pd.Series,
    held_positions: pd.Series,
    cost_bps: float,
    previous_position: float = 0.0,
) -> pd.DataFrame:
    turnover = held_positions.diff().abs()
    turnover.iloc[0] = abs(held_positions.iloc[0] - previous_position)
    costs = turnover * cost_bps / 10_000.0
    gross = held_positions * asset_returns
    return pd.DataFrame({
        "held_position": held_positions,
        "benchmark_return": asset_returns,
        "gross_return": gross,
        "turnover": turnover,
        "transaction_cost": costs,
        "net_return": gross - costs,
    })


def _sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0


def evaluate_ema_walk_forward(
    prices: pd.Series | pd.DataFrame,
    short_halflives: Iterable[float],
    long_halflives: Iterable[float],
    *,
    train_size: int = 1008,
    test_size: int = 504,
    n_splits: int = 6,
    warmup_size: int = 1600,
    transaction_cost_bps: float = 0.0,
    long_only: bool = True,
    periods_per_year: int = 252,
) -> WalkForwardResult:
    """Select by training net Sharpe, then freeze parameters for the next test.

    Sizes count observations (trading days for daily data). Each rolling training
    window has its own preceding EMA warm-up; neither warm-up nor test returns
    enter its selection score. Ties choose the smallest (short, long) pair.

    An end-of-day signal earns the following close-to-close return. Trading is
    modeled at that decision close, with costs booked against the next return;
    this is a simplified execution assumption. Training starts flat per candidate.
    Testing starts flat once, then carries holdings between contiguous folds,
    including turnover when parameters change. No terminal liquidation is charged.
    Benchmark returns cover exactly the test dates and are gross buy-and-hold.
    Sharpe uses a zero risk-free rate. Extra observations after the last requested
    fold are left unused; insufficient history raises rather than skipping folds.
    """
    for name, value in {
        "train_size": train_size, "test_size": test_size, "n_splits": n_splits,
        "warmup_size": warmup_size, "periods_per_year": periods_per_year,
    }.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if train_size < 2 or test_size < 2:
        raise ValueError("train_size and test_size must be at least 2")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and nonnegative")

    close = prices["Close"] if isinstance(prices, pd.DataFrame) else prices
    close = close.astype(float)
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    if close.index.hasnans or not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError("price dates must be unique, increasing, and nonmissing")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("prices must be finite, positive, and nonmissing")

    short_values = sorted(set(float(v) for v in short_halflives))
    long_values = sorted(set(float(v) for v in long_halflives))
    if any(not np.isfinite(v) or v <= 0 for v in short_values + long_values):
        raise ValueError("halflives must be finite and positive")
    candidates = [(s, l) for s, l in product(short_values, long_values) if s < l]
    if not candidates:
        raise ValueError("grid must contain a pair with short_halflife < long_halflife")
    required = warmup_size + train_size + n_splits * test_size
    if len(close) < required:
        raise ValueError(f"Need {required} price observations for {n_splits} folds; got {len(close)}")

    fold_rows, score_rows, daily_frames = [], [], []
    previous_position = 0.0
    for fold in range(n_splits):
        train_start = warmup_size + fold * test_size
        test_start = train_start + train_size
        test_end = test_start + test_size
        # Selection has no access to any price on or after test_start.
        training_prices = close.iloc[train_start - warmup_size:test_start]
        training_returns = training_prices.pct_change().iloc[warmup_size:]
        averages = {
            h: training_prices.ewm(halflife=h, adjust=False).mean()
            for h in sorted(set(short_values + long_values))
        }
        best_pair, best_score = candidates[0], -np.inf
        for short, long in candidates:
            targets = (averages[short] > averages[long]).astype(float)
            if not long_only:
                targets = 2.0 * targets - 1.0
            held = targets.shift(1).iloc[warmup_size:]
            training = _returns_for_positions(training_returns, held, transaction_cost_bps)
            score = _sharpe(training["net_return"], periods_per_year)
            score_rows.append({"fold": fold + 1, "short_halflife": short,
                               "long_halflife": long, "train_net_sharpe": score})
            if score > best_score:
                best_pair, best_score = (short, long), score

        short, long = best_pair
        # Causal EWM uses historical warm-up and updates during the test without
        # refitting parameters. The first test holding uses the last training close.
        history = close.iloc[train_start - warmup_size:test_end]
        fast = history.ewm(halflife=short, adjust=False).mean()
        slow = history.ewm(halflife=long, adjust=False).mean()
        targets = (fast > slow).astype(float)
        if not long_only:
            targets = 2.0 * targets - 1.0
        held = targets.shift(1).iloc[-test_size:]
        daily = _returns_for_positions(
            history.pct_change().iloc[-test_size:], held,
            transaction_cost_bps, previous_position,
        )
        previous_position = float(held.iloc[-1])
        daily["fold"] = fold + 1
        daily_frames.append(daily)
        metrics = compute_metrics(daily["net_return"], daily["benchmark_return"], periods_per_year)
        fold_rows.append({
            "fold": fold + 1,
            "warmup_start": training_prices.index[0],
            "warmup_end": close.index[train_start - 1],
            "train_start": close.index[train_start],
            "train_end": close.index[test_start - 1],
            "test_start": close.index[test_start],
            "test_end": close.index[test_end - 1],
            "n_train": train_size, "n_test": test_size,
            "short_halflife": short, "long_halflife": long,
            "train_net_sharpe": best_score,
            "transaction_cost_bps": transaction_cost_bps,
            "test_turnover": float(daily["turnover"].sum()),
            **{f"test_{key}": value for key, value in metrics.items()},
        })

    return WalkForwardResult(
        folds=pd.DataFrame(fold_rows),
        daily_returns=pd.concat(daily_frames).rename_axis("Date"),
        training_scores=pd.DataFrame(score_rows),
    )
