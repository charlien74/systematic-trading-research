from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


@dataclass
class CointegrationResult:
    intercept: float
    beta: float
    spread: pd.Series
    adf_statistic: float
    p_value: float


def fit_engle_granger(
    y: pd.Series,
    x: pd.Series,
) -> CointegrationResult:
    """
    Fit an Engle-Granger cointegration test.
    """
    data = pd.concat({"y": y, "x": x}, axis=1).dropna()

    dependent = data["y"]
    independent = data["x"]
    regressor_name = "x"

    x_design = sm.add_constant(independent)
    model = sm.OLS(dependent, x_design).fit()

    intercept = float(model.params["const"])
    beta = float(model.params[regressor_name])

    spread = dependent - intercept - beta * independent

    adf_statistic, p_value, *_ = adfuller(
        spread,
        regression="n",
        autolag="AIC",
    )

    return CointegrationResult(
        intercept=intercept,
        beta=beta,
        spread=spread,
        adf_statistic=float(adf_statistic),
        p_value=float(p_value),
    )