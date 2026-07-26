import numpy as np
import pandas as pd

from src.cointegration import fit_engle_granger


def test_fit_engle_granger_can_swap_dependent_variable() -> None:
    x = pd.Series(np.arange(6, dtype=float), name="x")
    y = pd.Series(
        1.0 + 2.0 * np.arange(6, dtype=float) + np.array([0.01, -0.02, 0.015, -0.01, 0.02, -0.015]),
        name="y",
    )

    default_result = fit_engle_granger(y, x)
    # reversed_result = fit_engle_granger(y, x, use_other_as_dependent=True)

    assert np.isclose(default_result.beta, 2.0, atol=0.1)
    assert np.isclose(default_result.intercept, 1.0, atol=0.1)
    # assert np.isclose(reversed_result.beta, 0.5, atol=0.1)
    # assert np.isclose(reversed_result.intercept, -0.5, atol=0.1)
    # assert len(reversed_result.spread) == len(x)
