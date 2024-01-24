"""
TODO:

- [x] serialise inputs:
    - [x] original signal
    - [x] unmixed_chromatograms
- [x] define target
    - [x] scores schema.

    
"""
from pandas.core.series import Series
import polars as pl

import hplc
import os
from typing import Any

from pandas.testing import assert_frame_equal
from hplc_py.fit_assessment import FitAssessment

from pandera.typing.pandas import DataFrame
import pytest

import pandas as pd

import pandas.testing as pdt

import pandera as pa
import pandera.typing as pt

import numpy as np
import numpy.typing as npt

from hplc_py.hplc_py_typing.hplc_py_typing import *
from hplc_py.hplc_py_typing.hplc_py_typing import PSignals

from hplc_py.quant import Chromatogram
import pickle

import hplc


def test_unmixed_df_exec(psignals: Any):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.relplot(psignals, x="time", y="amp_unmixed", hue="p_idx", kind="line")
    plt.show()

    pass


from hplc_py.hplc_py_typing.hplc_py_typing import PSignals
from hplc_py.map_signals.map_windows import WindowedSignal


class TestScores:
    @pytest.fixture
    def rtol(
        self,
    ):
        return 1e-2

    @pytest.fixture
    def ftol(self):
        return 1e-2

    @pytest.fixture
    def fa(
        self,
    ) -> FitAssessment:
        fa = FitAssessment()
        return fa

    @pytest.fixture
    def scores(
        self,
        fa: FitAssessment,
        psignals: DataFrame[PSignals],
        ws: DataFrame[WindowedSignal],
        rtol: float,
        ftol: float,
    ) -> pt.DataFrame:
        score_df = fa.calc_wdw_aggs(ws, psignals, rtol, ftol)

        return score_df

    def test_compare_windowed_signals(
        self,
        ws,
        main_window_df,
    ):
        ws = pl.from_pandas(ws).with_columns(source=pl.lit("mine"))
        mws = (
            main_window_df.rename(
                {
                    "signal_corrected": "amp",
                    "window_type": "w_type",
                    "window_id": "w_idx",
                }
            )
            .drop(
                "signal",
                "estimated_background",
            )
            .with_columns(source=pl.lit("main"))
        )
        df = pl.concat([ws, mws])
        df = (
            df.groupby(["source", "w_type", "w_idx"])
            .agg(start=pl.col("time_idx").first(), end=pl.col("time_idx").last())
            .sort(["start", "source"])
            .melt(
                id_vars=["source", "w_type", "w_idx"],
                value_name="time_idx",
                variable_name="bound",
            )
            .pivot(
                columns="source", values="time_idx", index=["w_type", "w_idx", "bound"]
            )
            .with_columns(
                diff=(pl.col("main") - pl.col("mine")).abs(),
                tol_perc=pl.lit(0.05),
                max=pl.max_horizontal("main", "mine"),
            )
            .with_columns(
                tol_act=pl.col("max") * (pl.col("tol_perc")),
            )
            .with_columns(
                tolpass=pl.col("diff") <= (pl.col("tol_act")),
            )
        )

        breakpoint()

    def test_score_df_exec(
        self,
        scores: DataFrame,
    ):
        print("")
        print(scores)

    def test_compare_scores(
        self,
        scores: pl.DataFrame,
        main_scores,
        ws,
        psignals,
    ):
        import polars.testing as polart
        import polars.selectors as cs
        import hvplot

        my_scores = scores
        main_scores = main_scores.collect().sort("window_type", "window_id")
        idx_cols = ["window_type", "window_id"]

        my_scores = (
            scores.rename(
                {
                    "w_type": "window_type",
                    "w_idx": "window_id",
                    "mixed_var": "signal_variance",
                    "mixed_mean": "signal_mean",
                    "mixed_fano": "signal_fano_factor",
                    "recon_score": "reconstruction_score",
                }
            )
            .select(
                pl.exclude(["tolpass", "u_peak_fano", "fano_div", "fanopass", "status"])
            )
            .sort("window_type", "window_id")
        )

        # melt
        my_scores = my_scores.melt(
            id_vars=idx_cols, variable_name="msnt", value_name="mine"
        )

        main_scores = main_scores.melt(
            id_vars=idx_cols, variable_name="msnt", value_name="main"
        )

        df = my_scores.join(
            main_scores, on=["window_type", "window_id", "msnt"], how="inner"
        )

        import polars.selectors as ps

        df = (
            df.with_columns(ps.float().round(10))
            .with_columns(
                diff=(pl.col("mine") - pl.col("main")).abs(),
                tol=pl.lit(0.05),
                hz_av=pl.sum_horizontal("mine", "main") / 2,
            )
            .with_columns(
                act_tol=pl.col("hz_av") * pl.col("tol"),
            )
            .with_columns(
                tolpass=pl.col("diff") <= pl.col("act_tol"),
            )
        )

        df = (
            df.select(pl.exclude("tolpass", "act_tol"))
            .fill_nan(0)
            .sort(
                "diff",
                descending=True,
            )
        )
        breakpoint()

    def test_psignals(
        self, main_chm_asschrom_fitted_pk, psignals, main_peak_window_recon_signal
    ):
        breakpoint()

    @pytest.fixture
    def fit_report(
        self,
        fa,
        ws,
        wum,
    ) -> DataFrame:
        fit_report = fa.assess_fit()
        return fit_report

    def test_fit_report_print(
        self,
        fa: FitAssessment,
        scores,
    ):
        scores_ = scores.to_pandas()
        fa.print_report_card(scores_)
