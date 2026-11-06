# Systematic Trading Research

A small exploratory Python project for researching and backtesting trend-following trading strategies. The current implementation focuses on simple exponential moving average (EMA) crossover rules and basic performance evaluation metrics such as return, volatility, and Sharpe ratio.

## What this project includes

- Price data loading and caching utilities for market data
- EMA indicator helpers
- A simple EMA crossover strategy for generating target positions
- Backtest helpers for computing strategy and benchmark returns
- Example notebooks and sample market data for experimentation

## Project structure

- [src/data/loader.py](src/data/loader.py) loads and caches OHLCV data from Yahoo Finance
- [src/indicators/ema.py](src/indicators/ema.py) computes EMA series
- [src/strategies/ema_crossover.py](src/strategies/ema_crossover.py) converts EMA signals into target positions
- [src/backtest.py](src/backtest.py) computes returns, volatility, Sharpe ratio, and cumulative performance
- [notebooks/](notebooks/) contains exploratory analysis notebooks
- [data/market/](data/market/) includes sample CSV datasets

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can also install the package in editable mode:

```bash
pip install -e .
```

## Quick start

The example below downloads price data, computes EMAs, builds a crossover signal, and evaluates the strategy:

```python
from src.data.loader import get_prices
from src.indicators.ema import get_ema_series
from src.strategies.ema_crossover import get_ema_crossover_positions
from src.backtest import compute_metrics_and_returns_from_positions_and_prices

prices = get_prices("AAPL", period="2y", auto_adjust=True)

fast_ema = get_ema_series(prices, period=20, halflife=10)
slow_ema = get_ema_series(prices, period=50, halflife=20)

positions = get_ema_crossover_positions(fast_ema, slow_ema)
metrics, strategy_curve, benchmark_curve = compute_metrics_and_returns_from_positions_and_prices(
    positions,
    prices,
)

print(metrics)
```

## Notes

- This repository is intentionally lightweight and research-oriented rather than production-focused.
- Data is cached locally under [data/market/](data/market/) to avoid repeated downloads.
- The notebooks in [notebooks/](notebooks/) are a good place to explore strategy behavior and experiment with parameters.

## Walk-forward evaluation

[src/walk_forward.py](src/walk_forward.py) implements rolling EMA parameter
selection on training data followed by evaluation with frozen parameters on the
next test window. The EMA notebook includes a six-fold NVDA experiment with
1,600 trading observations of warm-up, 1,008 training observations, and 504 test
observations per fold. Selection maximizes training Sharpe after a fixed 5 bps
cost per unit of turnover; Sharpe assumes a zero risk-free rate.

Run the same experiment offline from the repository root using the cached NVDA
CSV (or supply a compatible dated OHLCV CSV with `--prices`):

```bash
python -m scripts.run_ema_walk_forward
```

This writes `folds.csv`, `training_scores.csv`, `daily_returns.csv`, and
`experiment.json` under `results/ema_walk_forward/`. The fold table includes
training/test dates, selected half-lives, and net test metrics. Daily test returns
can be compounded into one continuous out-of-sample curve. The evaluator raises
if the requested folds do not fit instead of silently skipping them. Any history
after the last requested test is unused and counted in the saved metadata.

Signals at close t earn the close-to-close return at t+1 under a simplified
execution assumption; costs are booked against that following return. The test
portfolio starts flat once and carries positions between folds, paying turnover
when the selected model changes its allocation. No final liquidation is charged.
Buy-and-hold is a gross benchmark on identical test dates. Historical observations
initialize EMAs without contributing to training scores, and test observations
never enter their own fold's parameter selection. Later rolling training windows
can include earlier test data that has become available.

These are retrospective results on previously explored history and assets, not
an untouched holdout. The legacy period-grid and permutation plots remain
exploratory and are separate from this walk-forward evaluation.

Run the validation checks with `python -m pytest` (requires `pytest`).

## Roadmap

- Extend walk-forward evaluation to additional assets and reserve an untouched holdout.
- Test a strategy with a "flat channel" and see if this makes a long-short strategy viable.
- The same framework could be used to backtest other strategies, such as mean reversion and stat arb.

## License

This project is provided for research and educational purposes.

