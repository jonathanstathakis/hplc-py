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
    FloatArray
)
from hplc_py.map_signals.map_peaks import MapPeaks, PeakMap
from hplc_py.hplc_py_typing.interpret_model import interpret_model
from hplc_py.map_signals.map_windows import WindowedSignal, MapWindows

from hplc_py.deconvolve_peaks.mydeconvolution import DataPrepper, WdwPeakMap, InP0

import numpy as np

from typing import TypeAlias

Chromatogram: TypeAlias = None

chm=None

class TestDataPrepFix(TestMapPeaksFix):
    
    @pytest.fixture
    def mw(
        self,
    )->MapWindows:
        mw = MapWindows()
        return mw
    
    @pytest.fixture
    def left_bases(
        self,
        pm: DataFrame[PeakMap],
    )-> Series[pd.Int64Dtype]:
        left_bases: Series[pd.Int64Dtype] = Series[pd.Int64Dtype](pm[PeakMap.pb_left], dtype=pd.Int64Dtype())
        return left_bases
    
    @pytest.fixture
    def right_bases(
        self,
        pm: DataFrame[PeakMap],
    )-> Series[pd.Int64Dtype]:
        right_bases: Series[pd.Int64Dtype] = Series[pd.Int64Dtype](pm[PeakMap.pb_right], dtype=pd.Int64Dtype())
        return right_bases
    
    @pytest.fixture
    def amp(
        self,
        amp_bcorr: FloatArray,
    )-> Series[pd.Float64Dtype]:
        amp: Series[pd.Float64Dtype] = Series(amp_bcorr, name='amp', dtype=pd.Float64Dtype())
        return amp
    
    @pytest.fixture
    def ws(
        self,
        mw: MapWindows,
        time: Series[pd.Float64Dtype],
        amp: Series[pd.Float64Dtype],
        left_bases: Series[pd.Float64Dtype],
        right_bases: Series[pd.Float64Dtype],
    )->DataFrame[WindowedSignal]:
        
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
    )->DataFrame[PeakMap]:
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
    )->DataFrame[WdwPeakMap]:
        
        wpm = dp._window_peak_map(pm, ws)
        
        return wpm    
    
    @pytest.fixture
    def p0(
        self,
        dp: DataPrepper,
        wpm: DataFrame[PeakMap],
    ) -> DataFrame[P0]:
        
        wpm_ = wpm.loc[:,[InP0.window_idx, InP0.p_idx, InP0.amp, InP0.time, InP0.whh]]
        wpm_ = DataFrame[InP0](wpm_)
        p0 = dp._p0_factory(
            wpm_,
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
        
        default_bounds = dp._default_bounds_factory(
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
    )->DataFrame[Params]:
        
        params = dp._prepare_params(
            pm,
            ws,
            timestep
        )
        
        return params

class TestDataPrepper(TestDataPrepFix):
    
    def test_map_peaks_exec(
        self,
        pm: DataFrame[PeakMap],
    )->None:
        
        PeakMap.validate(pm, lazy=True)
        
    def test_window_peak_map(
        self,
        wpm: DataFrame[WdwPeakMap],
    )->None:
        
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
        
        
    
class TestDeconvolver:

    
    @pa.check_types
    @pytest.fixture
    def curve_fit_params(
        self,
        chm: Chromatogram,
        window: int,
        windowed_signal_df: DataFrame[WindowedSignal],
        my_param_df: DataFrame[Params],
    ):
        params = chm._deconvolve._prep_for_curve_fit(
            window,
            windowed_signal_df,
            "amp_corrected",
            my_param_df,
        )
        return params

    def test_prep_for_curve_fit(
        self,
        curve_fit_params: tuple[
            Series[float], Series[float], Series[float], Series[float], Series[float]
        ],
    ):
        pass

    @pa.check_types
    def test_popt_factory(
        self,
        popt_df: DataFrame[Popt],
        dset_schema,
    ):
        """
        TODO:
        - [ ] define dataset specific schemas
        - [ ] identify why algo needs more than 1200 iterations to minimize mine vs 33 for main
        - [ ] testing with the main adapted param_df, 24 iterations for the first window, 21 for the second. Whats the difference?

        Note: as of 2023-12-21 11:02:03 first window now takes 803 iterations. same window in main takes 70 iterations.
        """

        schema_tests(
            Popt,
            dset_schema,
            {"schema_name": "OutPoptDF_Base", "is_base": True},
            {"schema_name": "OutPoptDF_AssChrom", "inherit_from": "OutPoptDF_Base"},
            popt_df,
        )

        return None

    """
    2023-12-08 16:24:07

    Next is to..

    'assemble_deconvolved_peak_output'

    which includes:

    - for each peak, the optimum parameter:
        - amplitude
        - loc
        - whh
        - skew
        - area
        - and reconstructed signal.

    Both area and reconstructed signal are derived from `_compute_skewnorm` by passing
    the window time range and unpacking the optimized paramters.

    so we've already got the first 4. Need to construct the signal as a series, calculate its ara and add that to the popt df. We then construct a new frame where each column is a reconstructed signal series for each peak running the length of the original signal. The summation of that frame will provide the reconstructed convoluted signal for verification purposes.

    so, the reconstructed peak signal should have a peak_id and window_idx    
    """

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
    )->PeakDeconvolver:
        dp = PeakDeconvolver()
        return dp
    
    @pa.check_types
    def test_deconvolve_peaks(
        self,
        dp: PeakDeconvolver,
        ws: DataFrame[WindowedSignal]
    )->None:
        
        dp.deconvolve_peaks(
            ws,
        )
