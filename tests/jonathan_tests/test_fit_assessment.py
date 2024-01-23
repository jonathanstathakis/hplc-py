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


pd.options.display.max_columns = 50

pl.Config(set_tbl_cols=20)


@pytest.fixture
def main_fitted_chm():
    import hplc

    pkpth = "/Users/jonathan/hplc-py/tests/jonathan_tests/fitted_chm_main.pk"

    with open(pkpth, "rb") as f:
        main_fitted_chm = pickle.load(f)

    return main_fitted_chm


def adapt_ms_df(
    df: DataFrame,
):
    if not ("w_idx" in df.columns) & ("window_id" in df.columns):
        df = df.rename({"window_id": "w_idx"}, axis=1)

    ms_to_mys_mapping = {
        "signal_area": "area_amp_mixed",
        "inferred_area": "area_amp_unmixed",
        "signal_variance": "var_amp_unmixed",
        "reconstruction_score": "score",
        "signal_fano_factor": "mixed_fano",
    }

    df_ = df.rename(ms_to_mys_mapping, axis=1, errors="raise")

    df_["sw_idx"] = df_.groupby(["time_start"]).ngroup() + 1
    df_ = df_.loc[
        :,
        [
            "sw_idx",
            "w_type",
            "w_idx",
            "time_start",
            "time_end",
            "area_amp_mixed",
            "area_amp_unmixed",
            "var_amp_unmixed",
            "score",
            "mixed_fano",
            "applied_tolerance",
            "status",
        ],
    ]
    df_ = df_.sort_values("time_start").reset_index(drop=True)

    df_ = df_.reset_index(drop=True)
    return df_


@pytest.fixture
def m_sc_df(main_fitted_chm: hplc.quant.Chromatogram):
    ms_df = main_fitted_chm.assess_fit()
    adapted_ms_df = adapt_ms_df(ms_df)
    return adapted_ms_df


def test_ms_df_exec(m_sc_df: pd.DataFrame):
    pass


@pytest.fixture
def m_amp_recon(
    main_fitted_chm: hplc.quant.Chromatogram,
):
    """
    The peak_props dict does not contain information regarding x...
    """

    recon = pd.DataFrame(
        {p: v for p, v in enumerate(main_fitted_chm.unmixed_chromatograms)}
    )

    recon = recon.rename_axis(columns="time_idx", index="p_idx")
    recon = recon.T

    recon = pd.Series(
        np.sum(recon.to_numpy(), axis=1),
        name="amp_unmixed",
    )

    return recon


def test_m_amp_recon_exec(m_amp_recon: Series[Any]):
    pass


@pytest.fixture
def m_ws_df(
    main_fitted_chm: Chromatogram,
    m_amp_recon: DataFrame,
) -> DataFrame:
    m_ws_df = main_fitted_chm.window_df.copy(deep=True)

    m_ws_df = m_ws_df.rename(
        {
            "x": "time",
            "y_corrected": "amp_mixed",
            "window_id": "w_idx",
        },
        axis=1,
    )

    m_ws_df = m_ws_df.set_index(["w_type", "w_idx", "time_idx", "time"]).reset_index()
    m_ws_df = m_ws_df.drop(
        ["y", "estimated_background", "time_idx"], axis=1, errors="raise"
    )

    m_ws_df = pd.concat([m_ws_df, m_amp_recon], axis=1)

    m_ws_df = m_ws_df.astype(
        {
            "w_type": pd.StringDtype(),
            "w_idx": pd.Int64Dtype(),
            "time": pd.Float64Dtype(),
            "amp_mixed": pd.Float64Dtype(),
            "amp_unmixed": pd.Float64Dtype(),
        }
    )
    return m_ws_df


def test_main_w_df_exec(
    m_ws_df: DataFrame,
):
    pass


def test_signal_df_exec(signal_df: DataFrame):
    pass


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
    def path_fitted_chm(self):
        return os.path.join(os.getcwd(), "tests/jonathan_tests/fitted_chm.pk")

    @pytest.mark.skip
    def test_pickle_fitted_chm(
        self,
        fitted_chm: Chromatogram,
        path_fitted_chm: str,
    ):
        with open(path_fitted_chm, "wb") as f:
            pickle.dump(
                fitted_chm,
                f,
            )

    @pytest.fixture
    def fitted_chm_pk(
        self,
        path_fitted_chm: str,
    ):
        with open(path_fitted_chm, "rb") as f:
            fitted_chm = pickle.load(f)
        return fitted_chm

    @pytest.mark.xfail
    def test_chm_pickle(
        self,
        fitted_chm: Chromatogram,
        fitted_chm_pk: Chromatogram,
    ):
        assert fitted_chm == fitted_chm_pk

    @pytest.fixture
    def amp_col(self):
        return "amp_corrected"

    @pytest.fixture
    def fa(
        self,
    ) -> FitAssessment:
        fa = FitAssessment()
        return fa

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
            .melt(id_vars=['source','w_type','w_idx'], value_name='time_idx', variable_name='bound')
            .pivot(columns='source', values='time_idx', index=['w_type','w_idx','bound'])
            .with_columns(
                diff=(pl.col("main") - pl.col("mine")).abs(),
                tol_perc=pl.lit(0.05),
                max=pl.max_horizontal('main','mine'),
            )
            .with_columns(
                tol_act=pl.col("max")*(pl.col("tol_perc")),
            )
            .with_columns(
                tolpass=pl.col("diff")<=(pl.col("tol_act")),
            )
        )

        breakpoint()

    @pytest.fixture
    def fit_assess_wdw_aggs(
        self,
        fa: FitAssessment,
        psignals: DataFrame[PSignals],
        ws: DataFrame[WindowedSignal],
        rtol: float,
        ftol: float,
    ) -> pt.DataFrame:
        score_df = fa.calc_wdw_aggs(ws, psignals, rtol, ftol)

        return score_df

    def test_score_df_exec(
        self,
        fit_assess_wdw_aggs: DataFrame,
    ):
        print("")
        print(fit_assess_wdw_aggs)

    @pytest.fixture
    def m_sc_df(
        self,
        main_fitted_chm: Any,
    ):
        m_sc_df = main_fitted_chm._score_reconstruction()

        m_sc_df = m_sc_df.set_index(["w_type", "window_id"]).reset_index()

        m_sc_df = m_sc_df.rename({"window_id": "w_idx"}, axis=1)

        m_sc_df = m_sc_df.astype(
            {
                "w_type": pd.StringDtype(),
                "w_idx": pd.Int64Dtype(),
                **{
                    col: pd.Float64Dtype()
                    for col in m_sc_df
                    if pd.api.types.is_float_dtype(m_sc_df[col])
                },
            }
        )
        return m_sc_df

    def test_m_sc_df_exec(
        self,
        m_sc_df: DataFrame,
    ):
        print("")
        print(m_sc_df.dtypes)

    def test_compare_scores(
        self,
        fit_assess_wdw_aggs: pl.DataFrame,
        main_scores,
        ws,
        psignals,
    ):
        import polars.testing as polart
        import polars.selectors as cs
        import hvplot

        my_scores = fit_assess_wdw_aggs
        main_scores=main_scores.collect().sort('window_type','window_id')
        idx_cols = ["window_type", "window_id"]

        my_scores = (
            fit_assess_wdw_aggs
            .rename(
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
            ).sort('window_type','window_id')
        )
        
    
        # melt
        my_scores=my_scores.melt(id_vars=idx_cols, variable_name="msnt", value_name="mine")
        
        main_scores = main_scores.melt(
            id_vars=idx_cols, variable_name="msnt", value_name="main"
        )

        df = my_scores.join(main_scores, on=["window_type", "window_id", "msnt"], how="inner")

        import polars.selectors as ps
        df = (
            df
            .with_columns(ps.float().round(5))
            .with_columns(
                diff=(pl.col("mine") - pl.col("main")).abs(),
                tol=pl.lit(0.05),
                hz_av=pl.sum_horizontal('mine','main')/2,
                          )
            .with_columns(
                act_tol=pl.col('hz_av')*pl.col('tol'),
            )
            .with_columns(
                tolpass=pl.col('diff') <= pl.col('act_tol'),
            )
            # .filter(pl.col("tolpass") == False)
            # .sort('window_type','window_id','msnt')
        )
        
        breakpoint()


    def test_compare_window_dfs(
        self,
        ws,
        main_window_df,
    ):
        main_window_df = main_window_df.drop(["signal", "estimated_background"]).rename(
            {"window_type": "w_type", "window_id": "w_idx", "signal_corrected": "amp"}
        )
        ws = pl.from_pandas(ws)

        wdws = pl.concat(
            [
                ws.select(["w_type", "w_idx", "time_idx"]).with_columns(
                    source=pl.lit("mine")
                ),
                main_window_df.select(["w_type", "w_idx", "time_idx"]).with_columns(
                    source=pl.lit("main")
                ),
            ]
        )

        wdw_compare = (
            wdws.groupby(["source", "w_type", "w_idx"])
            .agg(start=pl.col("time_idx").first(), end=pl.col("time_idx").last())
            .sort(["start", "source"])
        )

        wdw_compare_diff = wdw_compare.pivot(
            columns="source", values=["start", "end"], index=["w_type", "w_idx"]
        ).with_columns(
            start_diff=pl.col("start_source_main")
            .sub(pl.col("start_source_mine"))
            .abs(),
            end_diff=pl.col("end_source_main").sub(pl.col("end_source_mine")).abs(),
        )

        breakpoint()
        import polars.testing as polt

        polt.assert_frame_equal(main_window_df.drop("estimated_background"), ws)
        breakpoint()

    def test_psignals(
        self, main_chm_asschrom_fitted_pk, psignals, main_peak_window_recon_signal
    ):
        breakpoint()

    def test_score_df_factory_exec(
        self,
        fit_assess_wdw_aggs: DataFrame,
    ) -> None:
        print(f"\n{fit_assess_wdw_aggs}")
        pass

    def test_ws_df_compare(
        self,
        ws_df: DataFrame,
        m_ws_df: DataFrame,
    ):
        left_df = ws_df
        right_df = m_ws_df
        try:
            assert_frame_equal(left_df, right_df)
        except Exception as e:
            err_str = str(e)
            err_str += "\n"

            cols = f"['left']: {left_df.columns}\n['right']: {right_df.columns}"

            err_str += cols
            err_str += "\n"

            dtypes = f"['left']: {left_df.dtypes}\n['right']: {right_df.dtypes}"
            err_str += "\n"
            err_str += dtypes

            raise AssertionError(err_str)

    @pytest.fixture
    def rtol(
        self,
    ):
        return 1e-2

    def test_assign_tolpass_exec(
        self,
        chm: Chromatogram,
        fit_assess_wdw_aggs: DataFrame,
        groups: list[str],
        rtol: float,
    ):
        chm._fitassess.assign_tolpass(fit_assess_wdw_aggs, groups, rtol)

    @pytest.fixture
    def main_fit_report(
        self,
        main_fitted_chm: hplc.quant.Chromatogram,
    ):
        report = main_fitted_chm.assess_fit()
        return report

    def test_main_fit_report_exec(
        self,
        main_fit_report: DataFrame,
    ):
        print("")
        print(main_fit_report)

    @pytest.fixture
    def ftol(self):
        return 1e-2

    @pytest.fixture
    def groups(self):
        return ["w_type", "w_idx"]

    def test_assign_fanopass_exec(
        self,
        chm: Chromatogram,
        fit_assess_wdw_aggs: DataFrame,
        groups: list[str],
        ftol: float,
    ):
        score_df_ = chm._fitassess.assign_fanopass(fit_assess_wdw_aggs, groups, ftol)

    @pytest.fixture
    def score_df_with_passes(
        self,
        chm: Chromatogram,
        fit_assess_wdw_aggs: DataFrame,
        groups: list[str],
        ftol: float,
        rtol: float,
    ):
        score_df_ = chm._fitassess.assign_tolpass(fit_assess_wdw_aggs, groups, rtol)
        score_df_ = chm._fitassess.assign_fanopass(fit_assess_wdw_aggs, groups, ftol)
        return score_df_

    def test_score_df_with_passes(
        self,
        score_df_with_passes: DataFrame,
    ):
        print("")
        print(score_df_with_passes)

    @pytest.fixture
    def score_df_with_status(
        self,
        chm: Chromatogram,
        score_df_with_passes: DataFrame,
    ):
        return chm._fitassess.assign_status(score_df_with_passes)

    def test_score_df_with_status(
        self,
        score_df_with_status: DataFrame,
    ):
        print("")
        print(score_df_with_status)

    @pytest.fixture
    def fit_report(
        self,
        chm: Chromatogram,
        signal_df: DataFrame,
        psignals: DataFrame,
        peak_report: DataFrame,
        window_df: DataFrame,
        rtol: float,
    ) -> DataFrame:
        fit_report = chm._fitassess.assess_fit(
            signal_df,
            psignals,
            window_df,
            rtol,
        )
        return fit_report

    def test_fit_report_exec(
        self,
        fit_report: DataFrame,
    ):
        pass

    def test_report_card_main(
        self,
        main_fitted_chm: hplc.quant.Chromatogram,
        fit_assess_wdw_aggs: DataFrame,
    ):
        main_fitted_chm.assess_fit()

    def test_termcolors(self):
        import termcolor

        termcolor.cprint("test", color="blue", on_color="white")
