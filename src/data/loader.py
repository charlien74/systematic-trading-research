"""Data loading utilities for market price series.

This module is intentionally small and notebook-friendly:
- pull OHLCV data from yfinance
- persist data under the project data directory
- reload cached CSVs into pandas DataFrames
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def project_root() -> Path:
	"""Return the repository root directory."""
	return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
	"""Return the top-level data directory and ensure it exists."""
	path = project_root() / "data"
	path.mkdir(parents=True, exist_ok=True)
	return path


def market_data_dir() -> Path:
	"""Return the directory where price datasets are stored."""
	path = data_dir() / "market"
	path.mkdir(parents=True, exist_ok=True)
	return path


def _normalize_ticker(ticker: str) -> str:
	return ticker.strip().upper().replace(" ", "")


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
	"""Return a stable single-index-column price frame for one ticker."""
	if isinstance(df.columns, pd.MultiIndex):
		if df.columns.nlevels == 2 and len(set(df.columns.get_level_values(1))) == 1:
			df = df.copy()
			df.columns = df.columns.get_level_values(0)
		else:
			raise ValueError("Expected single-ticker data but received multi-ticker columns")

	df = df.sort_index()
	df.index = pd.to_datetime(df.index)
	return df


def _sanitize_part(value: str) -> str:
	return value.replace("/", "-").replace(":", "-")


def _cache_scope(
	start: str | None = None,
	end: str | None = None,
	period: str | None = "max",
) -> str:
	if start is not None or end is not None:
		start_part = _sanitize_part(start) if start is not None else "open"
		end_part = _sanitize_part(end) if end is not None else "open"
		return f"start-{start_part}_end-{end_part}"

	period_value = "max" if period is None else str(period)
	return f"period-{_sanitize_part(period_value)}"


def _cache_file_name(
	ticker: str,
	start: str | None = None,
	end: str | None = None,
	period: str | None = "max",
	interval: str = "1d",
	auto_adjust: bool = True,
) -> str:
	normalized = _normalize_ticker(ticker).replace("/", "-")
	safe_interval = _sanitize_part(interval)
	scope = _cache_scope(start=start, end=end, period=period)
	adjust_part = "adj" if auto_adjust else "raw"
	return f"{normalized}_{scope}_{safe_interval}_{adjust_part}.csv"


def cache_path(
	ticker: str,
	start: str | None = None,
	end: str | None = None,
	period: str | None = "max",
	interval: str = "1d",
	auto_adjust: bool = True,
) -> Path:
	"""Return the canonical CSV cache path for a specific price request."""
	return market_data_dir() / _cache_file_name(
		ticker=ticker,
		start=start,
		end=end,
		period=period,
		interval=interval,
		auto_adjust=auto_adjust,
	)


def fetch_prices(
	ticker: str,
	start: str | None = None,
	end: str | None = None,
	period: str | None = "max",
	interval: str = "1d",
	auto_adjust: bool = True,
) -> pd.DataFrame:
	"""Download OHLCV history for one ticker from yfinance.

	Args:
		ticker: Yahoo Finance symbol, e.g. "SPY" or "AAPL".
		start: Optional start date (YYYY-MM-DD).
		end: Optional end date (YYYY-MM-DD).
		period: yfinance period (e.g. "1y", "5y", "max"). Ignored when
			both start and end are provided.
		interval: yfinance bar interval, default "1d".
		auto_adjust: Whether to return adjusted prices.

	Returns:
		A DataFrame indexed by DatetimeIndex with OHLCV columns.
	"""
	symbol = _normalize_ticker(ticker)

	# yfinance expects either explicit date bounds or a period.
	use_period = None if (start is not None or end is not None) else period
	df = yf.download(
		tickers=symbol,
		start=start,
		end=end,
		period=use_period,
		interval=interval,
		auto_adjust=auto_adjust,
		progress=False,
	)

	if df.empty:
		raise ValueError(f"No data returned for ticker '{symbol}'")

	# Keep a predictable shape for downstream strategy code.
	return _normalize_price_frame(df)


def fetch_prices_many(
	tickers: Iterable[str],
	start: str | None = None,
	end: str | None = None,
	period: str | None = "max",
	interval: str = "1d",
	auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
	"""Download data for multiple tickers and return a symbol->DataFrame map."""
	result: dict[str, pd.DataFrame] = {}
	for ticker in tickers:
		symbol = _normalize_ticker(ticker)
		result[symbol] = fetch_prices(
			ticker=symbol,
			start=start,
			end=end,
			period=period,
			interval=interval,
			auto_adjust=auto_adjust,
		)
	return result


def save_prices(
	df: pd.DataFrame,
	ticker: str,
	start: str | None = None,
	end: str | None = None,
	period: str | None = "max",
	interval: str = "1d",
	auto_adjust: bool = True,
	overwrite: bool = True,
) -> Path:
	"""Save a price DataFrame to the canonical local CSV cache path."""
	path = cache_path(
		ticker=ticker,
		start=start,
		end=end,
		period=period,
		interval=interval,
		auto_adjust=auto_adjust,
	)
	if path.exists() and not overwrite:
		raise FileExistsError(f"Cache file already exists: {path}")

	df.to_csv(path, index=True)
	return path


def load_cached_prices(
	ticker: str,
	start: str | None = None,
	end: str | None = None,
	period: str | None = "max",
	interval: str = "1d",
	auto_adjust: bool = True,
) -> pd.DataFrame:
	"""Load one cached ticker DataFrame from CSV."""
	path = cache_path(
		ticker=ticker,
		start=start,
		end=end,
		period=period,
		interval=interval,
		auto_adjust=auto_adjust,
	)
	if not path.exists():
		raise FileNotFoundError(f"No cached file found at: {path}")

	df = pd.read_csv(path, index_col=0)
	df.index = pd.to_datetime(df.index, errors="coerce")
	df = df[~df.index.isna()]
	df = _normalize_price_frame(df)
	return df


def get_prices(
	ticker: str,
	start: str | None = None,
	end: str | None = None,
	period: str | None = "max",
	interval: str = "1d",
	auto_adjust: bool = True,
	refresh: bool = False,
) -> pd.DataFrame:
	"""Return prices from cache if available, otherwise download and cache.

	Set refresh=True to force a new download.
	"""
	path = cache_path(
		ticker=ticker,
		start=start,
		end=end,
		period=period,
		interval=interval,
		auto_adjust=auto_adjust,
	)
	if path.exists() and not refresh:
		return load_cached_prices(
			ticker=ticker,
			start=start,
			end=end,
			period=period,
			interval=interval,
			auto_adjust=auto_adjust,
		)

	df = fetch_prices(
		ticker=ticker,
		start=start,
		end=end,
		period=period,
		interval=interval,
		auto_adjust=auto_adjust,
	)
	save_prices(
		df=df,
		ticker=ticker,
		start=start,
		end=end,
		period=period,
		interval=interval,
		auto_adjust=auto_adjust,
		overwrite=True,
	)
	return df


__all__ = [
	"project_root",
	"data_dir",
	"market_data_dir",
	"cache_path",
	"fetch_prices",
	"fetch_prices_many",
	"save_prices",
	"load_cached_prices",
	"get_prices",
]