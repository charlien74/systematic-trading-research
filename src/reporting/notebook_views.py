import pandas as pd
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def render_metrics_grouped(metrics : dict) -> None:
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]).rename_axis("Metric").reset_index()
    metrics_long = metrics_df.copy()
    parsed = metrics_long["Metric"].str.extract(r"^(strategy|benchmark)_(.+)$")
    metrics_long["side"] = parsed[0]
    metrics_long["metric"] = parsed[1]
    metrics_long = metrics_long.dropna(subset=["side", "metric"])

    metric_order = metrics_long["metric"].drop_duplicates().tolist()
    multi_cols = pd.MultiIndex.from_product([metric_order, ["benchmark", "strategy"]])

    metrics_grouped = metrics_long.assign(_row=0).pivot(
        index="_row",
        columns=["metric", "side"],
        values="Value",
    )
    metrics_grouped = metrics_grouped.reindex(columns=multi_cols)

    display(HTML(metrics_grouped.to_html(index=False)))

def plot_cumulative_returns(strategy_cumulative_returns : pd.Series,
                            benchmark_cumulative_returns : pd.Series,
                            title : str | None = None) -> None:
    plt.plot(strategy_cumulative_returns.index, strategy_cumulative_returns, label="Strategy Cumulative Returns")
    plt.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns, label="Benchmark Cumulative Returns")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gcf().autofmt_xdate()
    if title is not None:
        plt.title(title)
    plt.show()

def create_halflife_heatmaps(sharpe_array : np.ndarray, 
                             return_array : np.ndarray,
                             volatility_array : np.ndarray,
                             short_halflife_values : np.ndarray,
                             long_halflife_values : np.ndarray) -> None:
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(3, 1, figsize=(6, 16))
    sharpe_heatmap = ax[0].imshow(
        sharpe_array.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[
            short_halflife_values.min() - 2.5,
            short_halflife_values.max() + 2.5,
            long_halflife_values.min() - 2.5,
            long_halflife_values.max() + 2.5,
        ],
    )
    ax[0].set_title("Sharpe Ratio Across EMA Half-Life Pairs")
    ax[0].set_xlabel("Short Halflife")
    ax[0].set_ylabel("Long Halflife")

    return_heatmap = ax[1].imshow(
        return_array.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[
            short_halflife_values.min() - 2.5,
            short_halflife_values.max() + 2.5,
            long_halflife_values.min() - 2.5,
            long_halflife_values.max() + 2.5,
        ],
    )
    ax[1].set_title("Total Return Across EMA Half-Life Pairs")
    ax[1].set_xlabel("Short Halflife")
    ax[1].set_ylabel("Long Halflife")

    vol_heatmap = ax[2].imshow(
        volatility_array.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[
            short_halflife_values.min() - 2.5,
            short_halflife_values.max() + 2.5,
            long_halflife_values.min() - 2.5,
            long_halflife_values.max() + 2.5,
        ],
    )
    ax[2].set_title("Volatility Across EMA Half-Life Pairs")
    ax[2].set_xlabel("Short Halflife")
    ax[2].set_ylabel("Long Halflife")

    fig.colorbar(sharpe_heatmap, ax=ax[0], label="Sharpe Ratio")
    fig.colorbar(return_heatmap, ax=ax[1], label="Total Return")
    fig.colorbar(vol_heatmap, ax=ax[2], label="Volatility")
    plt.show()