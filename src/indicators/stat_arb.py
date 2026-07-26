import pandas as pd

import src.cointegration as coint


def compute_stat_arb_indicators(
    y: pd.Series,
    x: pd.Series,
    lookback: int = 252,
    train_step: int = 21,
) -> pd.DataFrame:
    """
    Use cointegration to generate indicators to be used to generate trading
    signals.
    """
    data = pd.concat({"y": y, "x": x}, axis=1).dropna()

    output = pd.DataFrame(
        index=data.index,
        columns=[
            "spread",
            "z_score",
            "hedge_ratio",
            "intercept",
            "coint_p_value",
        ],
        dtype=float,
    )

    n = len(data)

    for test_start in range(lookback, n, train_step):
        train_start = test_start - lookback
        test_end = min(test_start + train_step, n)

        train = data.iloc[train_start:test_start]
        test = data.iloc[test_start:test_end]

        fit = coint.fit_engle_granger(
            y=train["y"],
            x=train["x"],
        )

        train_spread = fit.spread
        spread_mean = train_spread.mean()
        spread_std = train_spread.std()


        # Apply frozen model parameters to the next out-of-sample block.
        test_spread = (
            test["y"]
            - fit.intercept
            - fit.beta * test["x"]
        )

        test_z_score = (test_spread - spread_mean) / spread_std
        

        output.loc[test.index, "spread"] = test_spread
        output.loc[test.index, "z_score"] = test_z_score
        output.loc[test.index, "beta"] = fit.beta
        output.loc[test.index, "intercept"] = fit.intercept
        output.loc[test.index, "coint_p_value"] = fit.p_value

    return output