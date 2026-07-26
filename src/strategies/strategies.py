import pandas as pd
import numpy as np


def get_ema_crossover_positions(
    fast_ema: pd.Series,
    slow_ema: pd.Series,
    long_only: bool = True) -> pd.Series:
    """
    Return target positions from a basic EMA crossover rule.
    """
    fast_ema, slow_ema = fast_ema.align(slow_ema, join="inner")

    bullish = fast_ema > slow_ema

    if long_only:
        positions = bullish.astype(float)
    else:
        positions = bullish.astype(float).replace({0.0: -1.0})

    positions.name = "target_position"
    return positions

def get_stat_arb_positions(
    indicators_df: pd.DataFrame,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    p_value_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Get stat-arb positions based on the indicators DF returned by 
    src.indicators.stat_arb.compute_stat_arb_indicators.
    The returned DataFrame will have columns for the spread position, as well
    as the individual positions for y and x.
    """
    required_columns = {"z_score", "coint_p_value", "beta"}
    missing_columns = required_columns - set(indicators_df.columns)

    if missing_columns:
        raise ValueError(
            "indicators_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    spread_positions = pd.Series(
        0.0,
        index=indicators_df.index,
        name="spread_position",
    )

    current_position = 0.0

    for timestamp, row in indicators_df.iterrows():
        z_score = row["z_score"]
        p_value = row["coint_p_value"]
        beta = row["beta"]

        eligible = (
            pd.notna(z_score)
            and pd.notna(p_value)
            and pd.notna(beta)
            and np.isfinite(z_score)
            and np.isfinite(p_value)
            and np.isfinite(beta)
            and p_value < p_value_threshold
        )

        # Close the trade if the model is currently invalid.
        if not eligible:
            current_position = 0.0

        # Open a new trade only while flat.
        elif current_position == 0.0:
            if z_score < -entry_threshold:
                current_position = 1.0   # long spread
            elif z_score > entry_threshold:
                current_position = -1.0  # short spread

        # Exit an existing trade once the spread returns near equilibrium.
        elif abs(z_score) < exit_threshold:
            current_position = 0.0

        spread_positions.loc[timestamp] = current_position

    beta = indicators_df["beta"]
    gross_exposure = 1.0 + beta.abs()

    positions = pd.DataFrame(index=indicators_df.index)
    positions["spread_position"] = spread_positions

    # spread = y - beta*x
    positions["y_position"] = (
        positions["spread_position"] / gross_exposure
    )
    positions["x_position"] = (
        -beta
        * positions["spread_position"]
        / gross_exposure
    )

    # Ensure rows with invalid beta do not create undefined exposures.
    positions[["x_position", "y_position"]] = (
        positions[["x_position", "y_position"]].fillna(0.0)
    )

    return positions