import pandas as pd
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
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
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gcf().autofmt_xdate()
    if title is not None:
        plt.title(title)
    plt.show()

def create_halflife_heatmaps(ema_grid_results: pd.DataFrame,
                             enforce_same_heatmap_scales: bool = False,
                             use_diverging_centered_scale: bool = False,
                             ) -> None:
    required_columns = {
        "ticker",
        "short_halflife",
        "long_halflife",
        "strategy_sharpe",
        "strategy_total_return",
        "strategy_annualized_volatility",
        "exposed_fraction",
    }
    missing_columns = required_columns.difference(ema_grid_results.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"ema_grid_results is missing required columns: {missing_list}")

    if use_diverging_centered_scale:
        cmap = LinearSegmentedColormap.from_list(
            "red_white_green",
            ["#8b0000", "#ffffff", "#006400"],
        )
    else:
        cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="white")

    short_halflife_values = np.sort(ema_grid_results["short_halflife"].unique())
    long_halflife_values = np.sort(ema_grid_results["long_halflife"].unique())
    tickers = ema_grid_results["ticker"].drop_duplicates().tolist()

    metric_specs = [
        ("strategy_sharpe", "Sharpe Ratio Across EMA Half-Life Pairs", "Sharpe Ratio"),
        ("strategy_total_return", "Total Return Across EMA Half-Life Pairs", "Total Return"),
        ("strategy_annualized_volatility", "Volatility Across EMA Half-Life Pairs", "Volatility"),
        ("exposed_fraction", "Exposed Fraction Across EMA Half-Life Pairs", "Exposed Fraction"),
    ]

    metric_color_ranges : dict[str, tuple[float, float]] = {}
    if enforce_same_heatmap_scales:
        for metric_column, _, _ in metric_specs:
            metric_values = ema_grid_results[metric_column].to_numpy(dtype=np.float32)
            if metric_column in ["strategy_annualized_volatility", "exposed_fraction"]:
                # Invert volatility and exposed fraction so lower values map to stronger red.
                metric_values = -metric_values
            if use_diverging_centered_scale:
                lim = float(np.nanmax(np.abs(metric_values)))
                metric_color_ranges[metric_column] = (-lim, lim)
            else:
                metric_color_ranges[metric_column] = (
                    float(np.nanmin(metric_values)),
                    float(np.nanmax(metric_values)),
                )

    fig, axes = plt.subplots(
        len(tickers),
        len(metric_specs),
        figsize=(6 * len(metric_specs), 5 * len(tickers)),
        squeeze=False,
    )

    extent = [
        short_halflife_values.min() - 2.5,
        short_halflife_values.max() + 2.5,
        long_halflife_values.min() - 2.5,
        long_halflife_values.max() + 2.5,
    ]

    for row_index, ticker in enumerate(tickers):
        ticker_results = ema_grid_results.loc[ema_grid_results["ticker"] == ticker]

        for column_index, (metric_column, title, colorbar_label) in enumerate(metric_specs):
            heatmap_values = (
                ticker_results
                .pivot(index="short_halflife", columns="long_halflife", values=metric_column)
                .reindex(index=short_halflife_values, columns=long_halflife_values)
                .to_numpy(dtype=np.float32)
            )

            if metric_column in ["strategy_annualized_volatility", "exposed_fraction"]:
                # Invert volatility and exposed fraction so lower values map to stronger red.
                heatmap_values = -heatmap_values

            axis = axes[row_index, column_index]
            imshow_kwargs = {}
            if enforce_same_heatmap_scales:
                vmin, vmax = metric_color_ranges[metric_column]
                imshow_kwargs = {"vmin": vmin, "vmax": vmax}
            elif use_diverging_centered_scale:
                lim = float(np.nanmax(np.abs(heatmap_values)))
                imshow_kwargs = {"vmin": -lim, "vmax": lim}

            heatmap = axis.imshow(
                heatmap_values.T,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                interpolation="nearest",
                extent=extent,
                **imshow_kwargs,
            )
            axis.set_title(f"{ticker}: {title}")
            axis.set_xlabel("Short Halflife")
            axis.set_ylabel("Long Halflife")
            fig.colorbar(heatmap, ax=axis, label=colorbar_label)

    plt.tight_layout()
    plt.show()

def plot_emas_and_price(data_df : pd.DataFrame, 
                        short_ema : pd.Series,
                        long_ema : pd.Series,
                        warmup_period : int,
                        title : str | None = None,
                        short_label : str | None = None,
                        long_label : str | None = None) -> None:

    plt.plot(data_df.index[warmup_period:], data_df['Close'].iloc[warmup_period:], color="tab:blue", label='Close Price')
    plt.plot(data_df.index[warmup_period:], short_ema, color="tab:orange", label=short_label)
    plt.plot(data_df.index[warmup_period:], long_ema, color="tab:green", label=long_label)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gcf().autofmt_xdate()

    plt.title(title if title is not None else "Close Price with Short and Long EMAs")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()
