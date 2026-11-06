import numpy as np
import pandas as pd
import pytest

from src.walk_forward import evaluate_ema_walk_forward


def prices():
    rng = np.random.default_rng(17)
    return pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, 70))),
        index=pd.bdate_range("2020-01-01", periods=70),
    )


def run(series, **kwargs):
    options = dict(train_size=20, test_size=10, n_splits=3, warmup_size=10,
                   transaction_cost_bps=5)
    options.update(kwargs)
    return evaluate_ema_walk_forward(series, [1, 2], [3, 5], **options)


def test_exact_disjoint_folds_and_matching_benchmark_dates():
    series = prices()
    result = run(series)
    assert len(result.folds) == 3
    assert len(result.daily_returns) == 30
    assert result.daily_returns.index.is_unique
    pd.testing.assert_index_equal(result.daily_returns.index, series.index[30:60], check_names=False)
    assert (result.folds.train_end < result.folds.test_start).all()
    assert (result.folds.test_end.iloc[:-1].to_numpy()
            < result.folds.test_start.iloc[1:].to_numpy()).all()
    np.testing.assert_allclose(result.daily_returns.benchmark_return, series.pct_change().iloc[30:60])


def test_test_prices_cannot_change_training_selection_or_first_test_holding():
    original = prices()
    changed = original.copy()
    changed.iloc[30:] *= np.exp(np.linspace(1, 5, len(changed) - 30))
    before, after = run(original), run(changed)
    pd.testing.assert_frame_equal(
        before.training_scores.query("fold == 1"),
        after.training_scores.query("fold == 1"),
    )
    columns = ["short_halflife", "long_halflife", "train_net_sharpe"]
    pd.testing.assert_series_equal(before.folds.iloc[0][columns], after.folds.iloc[0][columns])
    assert before.daily_returns.held_position.iloc[0] == after.daily_returns.held_position.iloc[0]
    # Changing later test prices also cannot alter earlier test holdings or returns.
    changed = original.copy()
    changed.iloc[35:] *= 2
    pd.testing.assert_frame_equal(run(original).daily_returns.iloc[:5], run(changed).daily_returns.iloc[:5])


def test_training_scores_use_only_training_returns_and_choose_maximum():
    series = prices()
    result = run(series)
    for row in result.training_scores.query("fold == 1").itertuples():
        history = series.iloc[:30]
        fast = history.ewm(halflife=row.short_halflife, adjust=False).mean()
        slow = history.ewm(halflife=row.long_halflife, adjust=False).mean()
        held = (fast > slow).astype(float).shift(1).iloc[10:]
        turnover = held.diff().abs()
        turnover.iloc[0] = abs(held.iloc[0])
        net = held * history.pct_change().iloc[10:] - turnover * 0.0005
        expected = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0
        assert row.train_net_sharpe == pytest.approx(expected)
    for fold in result.folds.itertuples():
        scores = result.training_scores.query("fold == @fold.fold")
        winner = scores.sort_values(
            ["train_net_sharpe", "short_halflife", "long_halflife"],
            ascending=[False, True, True],
        ).iloc[0]
        assert (fold.short_halflife, fold.long_halflife) == (winner.short_halflife, winner.long_halflife)


def test_execution_lag_and_turnover_across_fold_boundaries():
    series = prices()
    result = run(series, long_only=False)
    for fold in result.folds.itertuples():
        history = series.loc[fold.warmup_start:fold.test_end]
        fast = history.ewm(halflife=fold.short_halflife, adjust=False).mean()
        slow = history.ewm(halflife=fold.long_halflife, adjust=False).mean()
        expected = (2 * (fast > slow).astype(float) - 1).shift(1).loc[fold.test_start:]
        np.testing.assert_array_equal(
            result.daily_returns.query("fold == @fold.fold").held_position, expected,
        )
    daily = result.daily_returns
    turnover = daily.held_position.diff().abs()
    turnover.iloc[0] = abs(daily.held_position.iloc[0])
    np.testing.assert_allclose(daily.turnover, turnover)
    np.testing.assert_allclose(daily.net_return,
                               daily.held_position * daily.benchmark_return - turnover * 0.0005)


def test_ties_are_deterministic_and_costs_are_not_recharged_when_holdings_persist():
    series = prices()
    series[:] = np.arange(1, len(series) + 1)
    result = run(series)
    assert (result.folds.short_halflife == 1).all()
    assert (result.folds.long_halflife == 3).all()
    assert result.daily_returns.turnover.sum() == 1
    assert result.daily_returns.transaction_cost.sum() == pytest.approx(0.0005)


def test_insufficient_history_raises_instead_of_silently_skipping_folds():
    with pytest.raises(ValueError, match="Need 60 price observations"):
        run(prices().iloc[:59])
    assert len(run(prices().iloc[:60]).folds) == 3


@pytest.mark.parametrize("change", ["duplicates", "unordered", "missing", "zero"])
def test_invalid_prices_are_rejected(change):
    series = prices()
    if change == "duplicates":
        series.index = pd.DatetimeIndex([series.index[1], *series.index[1:]])
    elif change == "unordered":
        series = series.iloc[::-1]
    elif change == "missing":
        series.iloc[0] = np.nan
    else:
        series.iloc[0] = 0
    with pytest.raises(ValueError):
        run(series)
