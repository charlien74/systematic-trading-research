import pandas as pd
from IPython.display import HTML
import matplotlib.pyplot as plt

def render_metrics_grouped(metrics_df : pd.DataFrame) -> None:
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
    if title is not None:
        plt.title(title)
    plt.show()