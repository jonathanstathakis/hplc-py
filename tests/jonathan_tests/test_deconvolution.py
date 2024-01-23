import hvplot
from typing import Any, Callable, Literal, Tuple, TypeAlias

import numpy as np
import pandas as pd
import pandera as pa
import polars as pl
import pytest
from numpy import ndarray
from pandera.typing.pandas import DataFrame, Series
from pytest_benchmark.fixture import BenchmarkFixture

from hplc_py.deconvolve_peaks.mydeconvolution import (
    DataPrepper,
    InP0,
    PeakDeconvolver,
    WdwPeakMap,
)
from hplc_py.hplc_py_typing.hplc_py_typing import (
    P0,
    Bounds,
    FloatArray,
    OutPeakReportAssChrom,
    OutWindowDF_Base,
    Params,
    Popt,
    PReport,
    PSignals,
    SignalDF,
    schema_tests,
)
from hplc_py.hplc_py_typing.interpret_model import interpret_model
from hplc_py.map_signals.map_peaks import MapPeaks, PeakMap
from hplc_py.map_signals.map_windows import MapWindows, WindowedSignal
from tests.jonathan_tests.test_map_peaks import TestMapPeaksFix

Chromatogram: TypeAlias = None

chm = None


class TestDataPrepFix(TestMapPeaksFix):
    @pytest.fixture
    def wpm(
        self,
        dp: DataPrepper,
        pm: DataFrame[PeakMap],
        ws: DataFrame[WindowedSignal],
    ) -> DataFrame[WdwPeakMap]:
        wpm = dp._window_peak_map(pm, ws)

        return wpm

    @pytest.fixture
    def p0(
        self,
        dp: DataPrepper,
        wpm: DataFrame[PeakMap],
        timestep: float,
    ) -> DataFrame[P0]:
        wpm_ = wpm.loc[:, [InP0.w_idx, InP0.p_idx, InP0.amp, InP0.time, InP0.whh]]

        import pytest

        pytest.set_trace()

        wpm_ = DataFrame[InP0](wpm_)
        p0 = dp._p0_factory(
            wpm_,
            timestep,
        )

        return p0

    @pytest.fixture
    def default_bounds(
        self,
        dp: DataPrepper,
        p0: DataFrame[P0],
        ws: DataFrame[WindowedSignal],
        timestep: np.float64,
    ) -> DataFrame[Bounds]:
        default_bounds = dp._bounds_factory(
            p0,
            ws,
            timestep,
        )
        return default_bounds

    @pytest.fixture
    def params(
        self,
        dp: DataPrepper,
        pm: DataFrame[PeakMap],
        ws: DataFrame[WindowedSignal],
        timestep: np.float64,
    ) -> DataFrame[Params]:
        params = dp._prepare_params(pm, ws, timestep)

        return params


class TestDataPrepper(TestDataPrepFix):
    def test_map_peaks_exec(
        self,
        pm: DataFrame[PeakMap],
    ) -> None:
        PeakMap.validate(pm, lazy=True)

    def test_window_peak_map(
        self,
        wpm: DataFrame[WdwPeakMap],
    ) -> None:
        WdwPeakMap.validate(wpm, lazy=True)

    def test_p0_factory(
        self,
        p0: DataFrame[P0],
    ):
        """
        Test the initial guess factory output against the dataset-specific schema.
        """
        import pytest

        pytest.set_trace()
        P0.validate(p0, lazy=True)

    def test_default_bounds_factory(
        self,
        default_bounds: DataFrame[Bounds],
    ) -> None:
        """
        Define default bounds schemas
        """

        Bounds.validate(default_bounds, lazy=True)

    def test_prepare_params(
        self,
        dp: DataPrepper,
        params: DataFrame[Params],
    ) -> None:
        dp.prm_sc.validate(params, lazy=True)


class TestDeconvolverFix:
    @pytest.fixture
    def dc(
        self,
    ) -> PeakDeconvolver:
        dc = PeakDeconvolver()
        return dc

    @pytest.fixture
    def optimizer_jax(
        self,
    ):
        from jaxfit import CurveFit

        cf = CurveFit()
        return cf.curve_fit

    @pytest.fixture
    def optimizer_scipy(
        self,
    ):
        from scipy.optimize import curve_fit

        return curve_fit

    @pytest.fixture
    def popt_scipy(
        self,
        dc: PeakDeconvolver,
        ws: DataFrame[WindowedSignal],
        params: DataFrame[Params],
        optimizer_scipy: Callable[..., Any],
    ) -> DataFrame[Popt]:
        popt = dc._popt_factory(
            ws,
            params,
            optimizer_scipy,
            # optimizer_jax,
        )

        return popt

    @pytest.fixture
    def fit_func_jax(
        self,
    ):
        import hplc_py.skewnorms.skewnorms as sk

        return sk.fit_skewnorms_jax

    @pytest.fixture
    def popt(
        self,
        dc: PeakDeconvolver,
        ws: DataFrame[WindowedSignal],
        params: DataFrame[Params],
        optimizer_jax: Callable[..., Tuple[ndarray[Any, Any], ndarray[Any, Any]]],
        fit_func_jax: Callable[..., Any | Literal[0]],
    ) -> DataFrame[Popt]:
        popt = dc._popt_factory(
            ws,
            params,
            optimizer_jax,
            fit_func_jax,
        )

        return popt


class TestDeconvolver(TestDataPrepFix, TestDeconvolverFix):
    def test_popt_factory_benchmark(
        self,
        dc: PeakDeconvolver,
        ws: DataFrame[WindowedSignal],
        params: DataFrame[Params],
        optimizer_jax,
        fit_func_jax,
        benchmark: BenchmarkFixture,
    ):
        benchmark(
            dc._popt_factory,
            ws,
            params,
            optimizer_jax,
            fit_func_jax,
        )

    def test_popt_factory(
        self,
        popt: DataFrame[Popt],
    ):
        Popt.validate(popt, lazy=True)

        return None

    def test_store_popt(
        self,
        popt: DataFrame[Popt],
        popt_parqpath: Literal["/Users/jonathan/hplc-py/tests/jonathan_tests/assch…"],
    ):
        popt.to_parquet(popt_parqpath)

    def test_construct_peak_signals(self, psignals: DataFrame[PSignals]) -> None:
        PSignals.validate(psignals, lazy=True)

        return None

    @pytest.fixture
    def main_psignals(
        self,
        main_chm_asschrom_fitted_pk: Any,
    ):
        """
        The main package asschrom unmixed peak signals.

        Presented in long format in columns: ['time_idx','p_idx','main']
        """
        main_psignals = (
            pl.DataFrame(
                main_chm_asschrom_fitted_pk.unmixed_chromatograms,
                schema={
                    str(p): pl.Float64
                    for p in range(
                        main_chm_asschrom_fitted_pk.unmixed_chromatograms.shape[1]
                    )
                },
            )
            .with_row_index("time_idx")
            .melt(id_vars="time_idx", variable_name="p_idx", value_name="main")
            .with_columns(pl.col("p_idx").cast(pl.Int64))
        )

        return main_psignals

    def test_p_signals_compare_main(
        self,
        main_psignals: DataFrame,
        psignals: DataFrame[PSignals],
    ):
        import polars.selectors as ps
        psignals: pl.DataFrame = (
            pl.from_pandas(psignals).drop("time").rename({"amp_unmixed": "mine"})
        )

        signals = (
            psignals.with_columns(pl.col("time_idx").cast(pl.UInt32))
            .join(
                main_psignals,
                how="left",
                on=["p_idx", "time_idx"],
            )
            .select(["p_idx", "time_idx", "mine", "main"])
            .melt(
                id_vars=["p_idx", "time_idx"],
                value_vars=["mine", "main"],
                value_name="amp_unmixed",
                variable_name="source",
            )
            .with_columns(ps.float().round(8))
            .pivot(
                index=['p_idx','time_idx'], columns='source', values='amp_unmixed'
            )
            .with_columns(
                hz_av=pl.sum_horizontal('mine','main')/2,
                tol_perc=0.05,
                diff=(pl.col('mine')-pl.col('main')),
            )
            .with_columns(
                tol_act=pl.col('hz_av')*pl.col('tol_perc'),
            )
            .with_columns(
                tol_pass=pl.col('diff') <= pl.col('tol_act'),
            )
        )
        pass

    def test_reconstruct_signal(
        self,
        dc: PeakDeconvolver,
        psignals: DataFrame[PSignals],
    ):
        dc._reconstruct_signal(psignals)

    @pytest.fixture
    def p_signals_genned_on_subset(
        self,
        chm: Chromatogram,
        time: FloatArray,
        stored_popt: DataFrame[Popt],
        main_chm_asschrom_fitted_pk: Any,
        main_scores,
        main_params,
        main_popt,
        main_psignals,
        main_peak_report,
        main_windowed_peak_signals,
        main_peak_window_recon_signal,
    ):
        # test whether reconstruction based on the window time subset is equal to the reconstruction based on the whole time series within a tolerance of 10E-6

        # join main_psignals on main_peak_window_recon_signals to subset it then compare the two columns
        main_peak_window_recon_signal = main_peak_window_recon_signal.rename(
            {"reconstructed_signal": "amp_subset"}
        )

        # test if time_idx are the same
        wdwd = main_peak_window_recon_signal

        df = (
            wdwd.with_row_index()
            .filter(pl.col("time_idx").is_between(10000, 12000))
            .join(main_psignals, how="left", on=["p_idx", "time_idx"])
        )


        main_psignals = main_psignals.rename({"main": "amp_full"})
        df = main_peak_window_recon_signal.join(
            main_psignals, on=["time_idx"], how="left"
        )

        # hvplot.show(df.plot(x='time_idx',y=['amp_subset','amp_full']))
        # breakpoint()
        df: pl.DataFrame = df.with_columns(
            amp_diff=(pl.col("amp_subset") - pl.col("amp_full")).abs().round(6)
        )
        
        
        

        # import hvplot

        # hvplot.show(
        #     df.plot(
        #         x='time_idx',y=['amp_subset','amp_full'], groupby=['w_idx','p_idx'])
        # )

        breakpoint()


    def test_p_signals_genned_on_subset(self, p_signals_genned_on_subset: None):
        pass

    def test_peak_report(
        self,
        peak_report: DataFrame[PReport],
    ):
        PReport.validate(peak_report, lazy=True)

        return None

    def dp(
        self,
    ) -> PeakDeconvolver:
        dp = PeakDeconvolver()
        return dp

    @pa.check_types
    def test_deconvolve_peaks(
        self, dp: PeakDeconvolver, ws: DataFrame[WindowedSignal]
    ) -> None:
        dp.deconvolve_peaks(
            ws,
        )
