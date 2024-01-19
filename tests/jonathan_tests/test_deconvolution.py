import polars as pl

from tests.jonathan_tests.test_map_peaks import TestMapPeaksFix

import pandas as pd
import pandera as pa
import pytest
from pandera.typing.pandas import DataFrame, Series

from hplc_py.deconvolve_peaks.mydeconvolution import DataPrepper, PeakDeconvolver
from hplc_py.hplc_py_typing.hplc_py_typing import (
    Bounds,
    P0,
    Params,
    OutPeakReportAssChrom,
    OutPeakReportBase,
    OutWindowDF_Base,
    Popt,
    Recon,
    SignalDF,
    schema_tests,
    FloatArray,
)
from hplc_py.map_signals.map_peaks import MapPeaks, PeakMap
from hplc_py.hplc_py_typing.interpret_model import interpret_model
from hplc_py.map_signals.map_windows import WindowedSignal, MapWindows

from hplc_py.deconvolve_peaks.mydeconvolution import DataPrepper, WdwPeakMap, InP0

import numpy as np

from typing import TypeAlias

Chromatogram: TypeAlias = None

chm = None


class TestDataPrepFix(TestMapPeaksFix):
    @pytest.fixture
    def mw(
        self,
    ) -> MapWindows:
        mw = MapWindows()
        return mw

    @pytest.fixture
    def left_bases(
        self,
        pm: DataFrame[PeakMap],
    ) -> Series[pd.Int64Dtype]:
        left_bases: Series[pd.Int64Dtype] = Series[pd.Int64Dtype](
            pm[PeakMap.pb_left], dtype=pd.Int64Dtype()
        )
        return left_bases

    @pytest.fixture
    def right_bases(
        self,
        pm: DataFrame[PeakMap],
    ) -> Series[pd.Int64Dtype]:
        right_bases: Series[pd.Int64Dtype] = Series[pd.Int64Dtype](
            pm[PeakMap.pb_right], dtype=pd.Int64Dtype()
        )
        return right_bases

    @pytest.fixture
    def amp(
        self,
        amp_bcorr: FloatArray,
    ) -> Series[pd.Float64Dtype]:
        amp: Series[pd.Float64Dtype] = Series(
            amp_bcorr, name="amp", dtype=pd.Float64Dtype()
        )
        return amp

    @pytest.fixture
    def ws(
        self,
        mw: MapWindows,
        time: Series[pd.Float64Dtype],
        amp: Series[pd.Float64Dtype],
        left_bases: Series[pd.Float64Dtype],
        right_bases: Series[pd.Float64Dtype],
    ) -> DataFrame[WindowedSignal]:
        ws = mw.window_signal(
            left_bases,
            right_bases,
            time,
            amp,
        )

        return ws

    @pytest.fixture
    def pm(
        self,
        mp: MapPeaks,
        amp: Series[pd.Float64Dtype],
        time: Series[pd.Float64Dtype],
    ) -> DataFrame[PeakMap]:
        pm = mp.map_peaks(
            amp,
            time,
        )
        return pm

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
        wpm_ = wpm.loc[:, [InP0.window_idx, InP0.p_idx, InP0.amp, InP0.time, InP0.whh]]

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
        optimizer_scipy,
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
    def popt_jax(
        self,
        dc: PeakDeconvolver,
        ws: DataFrame[WindowedSignal],
        params: DataFrame[Params],
        optimizer_jax,
        fit_func_jax,
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
        benchmark,

    ):
        benchmark(dc._popt_factory,
            ws,
            params,
            optimizer_jax,
            fit_func_jax,
        )
        
    def test_popt_factory(
        self,
        popt_jax: DataFrame[Popt],

    ):
        Popt.validate(popt_jax, lazy=True)
        
        return None
    
    
    def test_unique_windows_benchmark(
        self,
        dc: PeakDeconvolver,
        ws: DataFrame[WindowedSignal],
        benchmark,
    ):
        
        
        
        def get_unique_windows(ws,):
            windows = ws.filter(pl.col('window_type')=='peak').select('window_idx').unique()
            return windows
        
        benchmark(get_unique_windows, ws)
    
    

    def test_reconstruct_peak_signal(
        self, unmixed_df: DataFrame[Recon], schema
    ) -> None:
        Recon.validate(unmixed_df, lazy=True)

        return None

    def test_peak_report(
        self,
        peak_report: DataFrame[OutPeakReportBase],
    ):
        schema_tests(
            OutPeakReportBase,
            OutPeakReportAssChrom,
            {
                "schema_name": "OutPeakReportBase",
                "is_base": True,
            },
            {
                "schema_name": "OutPeakReportAssChrom",
                "inherit_from": "OutPeakReportBase",
            },
            peak_report,
            verbose=False,
        )

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
