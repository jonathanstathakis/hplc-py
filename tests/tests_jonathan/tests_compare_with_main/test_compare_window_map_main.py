import pandas as pd
from numpy import float64
from numpy import int64
import numpy as np
import polars as pl
from hplc_py.map_signals.map_peaks.map_peaks import MapPeaks, PeakMap
from pandera.typing import Series, DataFrame
from hplc_py.hplc_py_typing.hplc_py_typing import WindowedSignal
import pytest


def test_compare_window_df_main(
    asschrom_ws: DataFrame[WindowedSignal],
    main_window_df: pd.DataFrame,
):
    """
    Compare windowing of the signal between the main and my implementation. Achieves this
    by joining the two together with columns - 'w_type', 'w_idx', and 't_idx' then finding
    the start and finish `t_idx` of each window, and after some reshaping, finds whether
    the absolute difference of the two sources for each window is less than 5% of the main value.
    If any of the windows has a difference greater than 5%, test will fail.
    """
    
    main_window_df = main_window_df.drop(["signal", "estimated_background"]).rename(
        {"window_type": "w_type", "window_id": "w_idx", "signal_corrected": "amp"}
    )
    ws_ = pl.from_pandas(asschrom_ws)

    wdws = pl.concat(
        [
            ws_.select(["w_type", "w_idx", "t_idx"]).with_columns(source=pl.lit("mine")),
            main_window_df.select(["w_type", "w_idx", "t_idx"]).with_columns(
                source=pl.lit("main")
            ),
        ]
    )

    wdw_compare = (
        wdws.group_by(["source", "w_type", "w_idx"])
        .agg(start=pl.col("t_idx").first(), end=pl.col("t_idx").last())
        .sort(["start", "source"])
        .melt(id_vars=["source", "w_type", "w_idx"])
        .pivot(columns="source", values="value", index=["w_type", "w_idx", "variable"])
        .with_columns(
            diff=pl.col("main").sub(pl.col("mine")).abs(),
            diff_tol=pl.col("main").mul(0.05),
        )
        .with_columns(tolpass=pl.col("diff_tol") >= pl.col("diff"))
    )

    assert wdw_compare.select('tolpass').to_series().all()
