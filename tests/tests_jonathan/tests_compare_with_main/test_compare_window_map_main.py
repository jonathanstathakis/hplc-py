import pandas as pd

import polars as pl
from pandera.typing import DataFrame
from hplc_py.map_windows.schemas import X_Windowed


def test_compare_window_df_main(
    asschrom_ws: DataFrame[X_Windowed],
    main_window_df: pd.DataFrame,
):
    """
    Compare windowing of the signal between the main and my implementation. Achieves this
    by joining the two together with columns - 'w_type', 'w_idx', and 't_idx' then finding
    the start and finish `t_idx` of each window, and after some reshaping, finds whether
    the absolute difference of the two sources for each window is less than 5% of the main value.
    If any of the windows has a difference greater than 5%, test will fail.
    """

    wdws_compare = (
        (
            pl.concat(
                [
                    asschrom_ws.pipe(pl.from_pandas).select(
                        ["w_type", "w_idx", "t_idx"], pl.lit("mine").alias("mine")
                    ),
                    main_window_df.drop(["signal", "estimated_background"])
                    .rename(
                        {
                            "window_type": "w_type",
                            "window_id": "w_idx",
                            "signal_corrected": "amp",
                        }
                    )
                    .pipe(pl.from_pandas)
                    .select(["w_type", "w_idx", "t_idx"])
                    .with_columns(source=pl.lit("main")),
                ]
            )
            .group_by(["source", "w_type", "w_idx"])
            .agg(start=pl.col("t_idx").first(), end=pl.col("t_idx").last())
            .sort(["start", "source"])
            .melt(id_vars=["source", "w_type", "w_idx"])
            .pivot(
                columns="source", values="value", index=["w_type", "w_idx", "variable"]
            )
            .with_columns(
                diff=pl.col("main").sub(pl.col("mine")).abs(),
                diff_tol=pl.col("main").mul(0.05),
            )
            .with_columns(tolpass=pl.col("diff_tol") >= pl.col("diff"))
        )
        .select("tolpass")
        .to_series()
        .all()
    )

    assert wdws_compare
