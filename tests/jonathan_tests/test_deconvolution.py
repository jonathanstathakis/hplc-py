from typing import Any, Callable, Literal, Tuple, TypeAlias

import numpy as np
import pandera as pa
import polars as pl
import polars.selectors as ps
import pytest
from numpy import float64, ndarray
from numpy.typing import NDArray
from pandera.typing.pandas import DataFrame
from pytest_benchmark.fixture import BenchmarkFixture

from hplc_py.deconvolve_peaks.mydeconvolution import (
    DataPrepper,
    InP0,
    PeakDeconvolver,
    WdwPeakMap,
    RSignal,
)
from hplc_py.hplc_py_typing.hplc_py_typing import (
    P0,
    Bounds,
    Params,
    Popt,
    PReport,
    PSignals,
    WindowedSignal,
)
from hplc_py.map_signals.map_peaks.map_peaks import PeakMap
from tests.jonathan_tests.test_map_peaks import TestMapPeaksFix

Chromatogram: TypeAlias = None

chm = None


class TestDataPrepFix(TestMapPeaksFix):
    @pytest.fixture
    def wpm(
        self,
        dp: DataPrepper,
        my_peak_map: DataFrame[PeakMap],
        ws: DataFrame[WindowedSignal],
    ) -> DataFrame[WdwPeakMap]:
        wpm = dp._window_peak_map(my_peak_map, ws)

        return wpm

    @pytest.fixture
    def p0(
        self,
        dp: DataPrepper,
        wpm: DataFrame[PeakMap],
        timestep: float64,
    ) -> DataFrame[P0]:
        wpm_ = wpm.loc[:, [InP0.w_idx, InP0.p_idx, InP0.amp, InP0.time, InP0.whh]]

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
        timestep: float64,
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
        my_peak_map: DataFrame[PeakMap],
        ws: DataFrame[WindowedSignal],
        timestep: float64,
    ) -> DataFrame[Params]:
        params = dp._prepare_params(my_peak_map, ws, timestep)

        return params


class TestDataPrepper(TestDataPrepFix):
    def test_map_peaks_exec(
        self,
        my_peak_map: DataFrame[PeakMap],
    ) -> None:
        PeakMap.validate(my_peak_map, lazy=True)

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
    def fit_func_scipy(self):
        import hplc_py.skewnorms.skewnorms as sk

        return sk._fit_skewnorms_scipy

    @pytest.fixture
    def popt_scipy(
        self,
        dc: PeakDeconvolver,
        ws: DataFrame[WindowedSignal],
        params: DataFrame[Params],
        optimizer_scipy: Callable[..., Any],
        fit_func_scipy: Callable,
    ) -> DataFrame[Popt]:
        popt = dc._popt_factory(
            ws,
            params,
            optimizer_scipy,
            fit_func_scipy,
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
        main_psignals: pl.DataFrame,
        psignals: DataFrame[PSignals],
    ):
        psignals_: pl.DataFrame = (
            pl.from_pandas(psignals).drop("time").rename({"amp_unmixed": "mine"})
        )

        signals = (
            (
                psignals_.with_columns(pl.col("time_idx").cast(pl.UInt32))
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
                    index=["p_idx", "time_idx"], columns="source", values="amp_unmixed"
                )
                .with_columns(
                    hz_av=pl.sum_horizontal("mine", "main") / 2,
                    tol_perc=0.05,
                    diff=(pl.col("mine") - pl.col("main")),
                )
                .with_columns(
                    tol_act=pl.col("hz_av") * pl.col("tol_perc"),
                )
                .with_columns(
                    tol_pass=pl.col("diff") <= pl.col("tol_act"),
                )
            )
            .group_by("p_idx")
            .agg(
                perc_True=pl.col("tol_pass").filter(tol_pass=True).count().truediv(pl.col("tol_pass").count()).mul(100),
                perc_False=pl.col("tol_pass").filter(tol_pass=False).count().truediv(pl.col("tol_pass").count()).mul(100),
            )
            .sort("p_idx")
        )

        breakpoint()
        import polars.testing as pt
        pt.assert_frame_equal(signals.select((pl.col('perc_False')<5).all()), pl.DataFrame({"perc_False":True}))

    @pa.check_types
    def test_reconstruct_signal(
        self,
        r_signal: DataFrame[RSignal],
    ):
        pass
        

    @pytest.fixture
    def p_signals_genned_on_subset(
        self,
        chm: Chromatogram,
        time: NDArray[float64],
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

    def test_p_signals_genned_on_subset(self, p_signals_genned_on_subset: None):
        pass

    def test_peak_report(
        self,
        peak_report: DataFrame[PReport],
    ):
        PReport.validate(peak_report, lazy=True)

        return None

    def dc(
        self,
    ) -> PeakDeconvolver:
        dc = PeakDeconvolver()
        return dc

    @pa.check_types
    def test_deconvolve_peaks(
        self,
        dc: PeakDeconvolver,
        ws: DataFrame[WindowedSignal],
        my_peak_map: DataFrame[PeakMap],
        timestep: float64,
    ) -> None:
        dc.deconvolve_peaks(
            my_peak_map,
            ws,
            timestep,
        )
