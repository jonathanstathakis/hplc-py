"""
Best Practices:
    - Define Fixture classes which are inherited into Test classes.
    - Define stateless method classes which provide methods for a master state storage class.

"""
import hplc
from typing import Any, Literal, cast, Optional
from matplotlib.figure import Figure
from pandas.core.arrays.base import ExtensionArray
from pandera.typing.pandas import DataFrame, Series
import pytest
import typing

import pandas as pd
import pandera as pa
import pandera.typing as pt

import numpy as np
from numpy import float64, floating
import numpy.typing as npt
from numpy.typing import NDArray

import matplotlib.pyplot as plt


import matplotlib as mpl

from scipy import integrate  # type: ignore


import copy


from hplc_py.quant import Chromatogram

from hplc_py.deconvolve_peaks.mydeconvolution import DataPrepper, PPeakDeconvolver
from hplc_py.baseline_correct.correct_baseline import CorrectBaseline, SignalDFBCorr
from hplc_py.map_signals.map_signals import (
    MapPeaks,
    WHH,
    PeakBases,
    MapPeakPlots,
    MapPeaksMixin,
    MapWindows,
    FindPeaks,
    PeakMap,
    PeakWindows,
    PWdwdTime,
    IPBounds,
    WindowedTime,
    WindowedSignalDF
)


from .conftest import AssChromResults

from hplc_py.hplc_py_typing.hplc_py_typing import (
    SignalDFInBase,
    OutSignalDF_Base,
    OutWindowDF_Base,
    OutWindowDF_AssChrom,
    OutInitialGuessBase,
    OutInitialGuessAssChrom,
    OutDefaultBoundsBase,
    OutDefaultBoundsAssChrom,
    OutWindowedSignalBase,
    OutWindowedSignalAssChrom,
    OutParamsBase,
    OutParamsAssChrom,
    OutPoptDF_Base,
    OutPoptDF_AssChrom,
    OutReconDFBase,
    OutReconDF_AssChrom,
    OutPeakReportBase,
    OutPeakReportAssChrom,
    interpret_model,
    schema_tests,
    FloatArray,
    IntArray,
)


OutPeakDFAssChrom = PeakMap

from pandas.testing import assert_frame_equal

pd.options.display.precision = 9
pd.options.display.max_columns = 50


class TestInterpretModel:
    def schema_cls(
        self,
        schema_str,
    ):
        # instantiate the schema class
        exec(schema_str)

        # get the schema class object from locals
        schema_cls = locals()["InSampleDF"]

        return schema_cls

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [-1, -2, -3, -4, -5],
            }
        )

    @pytest.fixture
    def test_gen_sample_df_schema(self, sample_df: DataFrame):
        interpret_model(sample_df)

        return None

    @pytest.fixture
    def eq_schema_str(
        self,
        sample_df: DataFrame,
    ):
        check_dict = {col: "eq" for col in sample_df.columns}
        schema_def_str = interpret_model(sample_df, "InSampleDF", "", check_dict)

        return schema_def_str

    @pytest.fixture
    def isin_schema_str(
        self,
        sample_df: DataFrame,
    ):
        check_dict = {col: "isin" for col in sample_df.columns}
        schema_def_str = interpret_model(sample_df, "InSampleDF", "", check_dict)

        return schema_def_str

    @pytest.fixture
    def basic_stats_schema_str(self, sample_df: DataFrame):
        numeric_cols = []
        non_numeric_cols = []
        for col in sample_df:
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)

        # test numeric cols with 'basic_stats'
        numeric_col_check_dict = dict(
            zip(numeric_cols, ["basic_stats"] * len(numeric_cols))
        )
        non_numeric_col_check_dict = dict(
            zip(non_numeric_cols, ["isin"] * len(non_numeric_cols))
        )

        check_dict = dict(**numeric_col_check_dict, **non_numeric_col_check_dict)
        schema_def_str = interpret_model(sample_df, "InSampleDF", "", check_dict)

        return schema_def_str

    def test_eq_schema(
        self,
        sample_df: DataFrame,
        eq_schema_str: str,
    ):
        """
        Test whether the 'equals' schema works as expected
        """
        schema = self.schema_cls(eq_schema_str)
        schema(sample_df)

    def test_isin_schema(
        self,
        sample_df: DataFrame,
        isin_schema_str: str,
    ):
        """
        Test whether the 'isin' schema works as expected
        """
        schema = self.schema_cls(isin_schema_str)
        schema(sample_df)

    def test_basicstats_schema(
        self,
        sample_df: DataFrame,
        basic_stats_schema_str: str,
    ):
        """
        Test whether the 'basic_stats' schema works as expected
        """

        schema = self.schema_cls(basic_stats_schema_str)
        schema(sample_df)

    def test_base_schema(self, sample_df: DataFrame):
        # schema = self.schema_cls(sample_df)
        interpret_model(sample_df, "TestBaseSchema", is_base=True)


def schema_error_str(schema, e, df):
    err_str = "ERROR REPORT:"
    err_str += "ERROR: " + str(e) + "\n"
    err_str += "SCHEMA: " + str(schema) + "\n"
    err_str += "ACTUALS:\n"
    for col in df:
        err_str += str(col) + "\n"
        err_str += str(df[col].tolist()) + "\n"

    err_str += "compare these against the schema and replace where necessary"
    raise RuntimeError(err_str)


def schema_error_str_long_frame(schema, e, df):
    err_str = "ERROR REPORT:"
    err_str += "ERROR: " + str(e) + "\n"
    err_str += "SCHEMA: " + str(schema) + "\n"
    err_str += "ACTUALS:\n"
    for col in df:
        err_str += str(col) + "\n"
        col_min = df[col].min()
        col_max = df[col].max()
        err_str += f"{{'min_value':{col_min}, 'max_value':{col_max}}}" + "\n"

    err_str += "compare these against the schema and replace where necessary"
    raise RuntimeError(err_str)


manypeakspath = "/Users/jonathan/hplc-py/tests/test_data/test_many_peaks.csv"
asschrompath = "tests/test_data/test_assessment_chrom.csv"


def test_acr(acr: AssChromResults):
    assert acr
    pass


def test_param_df_adapter(acr: AssChromResults):
    param_df = acr.tables["asschrom_param_tbl"]

    adapted_param_df = acr.adapt_param_df(param_df)
    try:
        OutParamsBase(adapted_param_df)
    except Exception as e:
        raise RuntimeError(e)


def check_df_exists(df):
    assert isinstance(df, pd.DataFrame)
    assert df.all
    return None


def test_target_window_df_exists(target_window_df: Any):
    check_df_exists(target_window_df)
    pass


def test_get_asschrom_results(acr: AssChromResults):
    """
    Simply test whether the member objects of AssChromResults are initialized.
    """

    for tbl in acr.tables:
        check_df_exists(acr.tables[tbl])


def test_timestep_exists_and_greater_than_zero(timestep: float):
    assert timestep
    assert timestep > 0


def test_amp_raw_not_null(amp_raw):
    """
    for exploring shape and behavior of amp. a sandpit
    """
    assert all(amp_raw)


class TestTimeStep:
    def test_timestep(self, timestep: float):
        assert timestep


class TestLoadData:
    def test_loaded_signal_df(
        self,
        loaded_signal_df: DataFrame[OutSignalDF_Base],
    ):
        loaded_signal_df = cast(pd.DataFrame, loaded_signal_df)

        if not isinstance(loaded_signal_df, pd.DataFrame):
            raise TypeError("loaded_signal_df expected to be castable to pd.DataFrame")

        if loaded_signal_df.empty:
            raise ValueError("loaded_signal_df is expected to be populated")


@pytest.mark.skip
class TestTimeWindows:
    @pytest.fixture
    def valid_time_windows(self):
        return [[0, 5], [5, 15]]

    @pytest.fixture
    def invalid_time_window(self):
        return [[15, 5]]

    @pytest.mark.skip
    def test_crop_valid_time_windows(
        self,
        chm: Chromatogram,
        in_signal: pt.DataFrame[SignalDFInBase],
        valid_time_windows: list[list[int]],
    ):
        """
        test `crop()` by applying a series of valid time windows then testing whether all values within the time column fall within that defined range.
        """

        for window in valid_time_windows:
            assert len(window) == 2

            # copy to avoid any leakage across instances

            in_signal = chm.crop(in_signal, time_window=window)

            leq_mask = in_signal.time >= window[0]
            geq_mask = in_signal.time <= window[1]

            assert (leq_mask).all(), f"{in_signal[leq_mask].index}"
            assert (geq_mask).all(), f"{in_signal[geq_mask].index}"

        return None

    @pytest.mark.skip
    def test_crop_invalid_time_window(
        self,
        chm: Chromatogram,
        in_signal: pt.DataFrame[SignalDFInBase],
        invalid_time_window: list[list[int]],
    ):
        for window in invalid_time_window:
            try:
                chm.crop(in_signal, window)

            except RuntimeError as e:
                continue


"""
TODO:

add comprehensive tests for `load_data`
"""

"""
2023-11-27 06:26:22

test `correct_baseline`
"""


class TestCorrectBaseline:
    @pytest.fixture
    def debug_bcorr_df_main(self, acr: AssChromResults):
        """
        ['timestep', 'shift', 'n_iter', 'signal', 's_compressed', 's_compressed_prime', 'inv_tform', 'y_corrected', 'background']
        """
        return acr.tables["bcorr_dbug_tbl"]

    @pytest.fixture
    def target_s_compressed_prime(
        self,
        debug_bcorr_df_main: Any,
    ):
        return debug_bcorr_df_main["s_compressed_prime"].to_numpy(np.float64)

    @pytest.fixture
    def windowsize(self):
        return 5

    @pytest.fixture
    def amp_clipped(
        self,
        cb: CorrectBaseline,
        amp_raw: FloatArray,
    ) -> FloatArray:
        return cb.shift_and_clip_amp(amp_raw)

    @pytest.fixture
    def s_compressed(self, cb: CorrectBaseline, amp_clipped: FloatArray) -> FloatArray:
        # intensity raw compressed
        s_compressed = cb.compute_compressed_signal(amp_clipped)

        return s_compressed

    @pytest.fixture
    def s_compressed_main(
        self,
        acr: AssChromResults,
    ):
        s_compressed = acr.tables["bcorr_dbug_tbl"]["s_compressed"]

        return s_compressed

    def test_s_compressed_against_main(
        self,
        s_compressed: NDArray[float64],
        s_compressed_main: Any,
    ):
        """
        test calculated s_compressed against the main version
        """
        assert all(np.equal(s_compressed, s_compressed_main))

    def test_s_compressed_against_dbug_tbl(
        self,
        s_compressed: FloatArray,
        debug_bcorr_df: pd.DataFrame,
    ):
        """ """

        diff_tol = 5e-10

        s_comp_df = pd.DataFrame(
            {
                "main": debug_bcorr_df.s_compressed,
                "mine": s_compressed,
                "diff_tol": diff_tol,
            }
        )

        s_comp_df["diff"] = s_comp_df["main"] - s_comp_df["mine"]

        s_comp_df["is_diff"] = s_comp_df["diff"].abs() > s_comp_df["diff_tol"]

        if s_comp_df["is_diff"].any():
            plt.plot(s_compressed, label="s_compressed")
            plt.plot(debug_bcorr_df.s_compressed, label="debug tbl")
            plt.suptitle("Divergent S compressed series")
            plt.legend()
            plt.show()
            raise ValueError(
                f"my s_compressed is divergent from main s_compressed in {(s_comp_df['is_diff']==True).shape[0]} elements above a threshold of {diff_tol}"
            )

    def test_amp_raw_equals_main(
        self,
        amp_raw: FloatArray,
        amp_raw_main: FloatArray,
    ):
        assert np.all(np.equal(amp_raw, amp_raw_main))

    def test_amp_compressed_exists_and_is_array(
        self,
        s_compressed: NDArray[float64],
    ):
        assert np.all(s_compressed)
        assert isinstance(s_compressed, np.ndarray)

    @pytest.fixture
    def n_iter(
        self,
        chm: Chromatogram,
        windowsize: Literal[5],
        timestep: float,
    ):
        return chm._baseline.compute_n_iter(windowsize, timestep)

    @pytest.fixture
    def s_compressed_prime(
        self,
        cb: CorrectBaseline,
        s_compressed: NDArray[float64],
        n_iter: int,
    ):
        s_compressed_prime = cb.compute_s_compressed_minimum(
            s_compressed,
            n_iter,
        )
        return s_compressed_prime

    def test_s_compressed_prime_set(self, s_compressed_prime: FloatArray):
        pass

    def test_amp_compressed_prime_against_main(
        self,
        s_compressed_prime: FloatArray,
        target_s_compressed_prime: FloatArray,
    ):
        if not np.all(s_compressed_prime == target_s_compressed_prime):
            raise ValueError("`amp_compressed_prime` does not equal target")
        return None

    def test_compute_inv_tfrom(
        self,
        chm: Chromatogram,
        amp_raw: FloatArray,
    ) -> None:
        chm._baseline.compute_inv_tform(amp_raw)

    def test_correct_baseline(
        self,
        loaded_cb: CorrectBaseline,
        amp_bcorr_colname: str,
        time_colname: str,
        amp_raw: FloatArray,
    ) -> None:
        # pass the test if the area under the corrected signal is less than the area under the raw signal

        signal_df = loaded_cb.correct_baseline()

        amp_bcorr = signal_df[amp_bcorr_colname].to_numpy(np.float64)
        time = signal_df[time_colname].to_numpy(np.float64)

        x_start = time[0]
        x_end = time[-1]
        n_x = len(time)

        # add a preset baseline to ensure that correction behavior is within expected scale

        x = np.linspace(x_start, x_end, n_x)

        from scipy import stats

        skew = 1
        loc = x_end * 0.3
        scale = x_end * 0.3
        skewnorm = stats.skewnorm(skew, loc=loc, scale=scale)

        y = skewnorm.pdf(x) * np.max(amp_raw) ** 2

        added_baseline = amp_raw + y

        baseline_auc = integrate.trapezoid(added_baseline, time)

        bcorr_auc = integrate.trapezoid(amp_bcorr, time)

        assert baseline_auc > bcorr_auc

    @pytest.fixture
    def debug_bcorr_df(
        self,
        bcorred_cb: CorrectBaseline,
    ):
        debug_df = bcorred_cb._debug_bcorr_df

        return debug_df

    @pytest.mark.skip
    def test_debug_bcorr_df_compare_s_compressed_prime(
        self, s_compressed_prime: NDArray[float64], debug_bcorr_df: DataFrame
    ):
        """
        I am expecting these two series to be identical, however they currently are not. the debug df is the same as the target.
        """
        diff_tol = 1e-10
        prime_df = pd.DataFrame(
            {
                "mine": s_compressed_prime,
                "main": debug_bcorr_df["s_compressed_prime"],
                "diff_tol": diff_tol,
            }
        )

        prime_df["diff"] = prime_df["main"] - prime_df["mine"]

        prime_df["is_diff"] = prime_df["diff"].abs() > prime_df["diff_tol"]

        if prime_df["is_diff"].any():
            plt.plot(debug_bcorr_df["s_compressed_prime"], label="debug series")
            plt.plot(s_compressed_prime, label="isolated")
            plt.legend()
            plt.suptitle("divergent s_comp_prime series")
            plt.show()

            raise ValueError(
                f"Divergence greater than {diff_tol} detected between my s_compressed_prime and main over {(prime_df['is_diff']==True).shape[0]} elements"
            )

    def test_compare_timestep(self, timestep: float, debug_bcorr_df: DataFrame):
        difference = timestep - debug_bcorr_df.iloc[0]

        return None


class TestMapPeaksFix:
    
    @pytest.fixture
    def mpm(
        self,
    ) -> MapPeaksMixin:
        mpm = MapPeaksMixin()

        return mpm
    
    @pytest.fixture
    def mp(
        self,
    )->MapPeaks:
        mp = MapPeaks()
        return mp

    @pytest.fixture
    def prom(self) -> float:
        return 0.01

    @pytest.fixture
    def wlen(self) -> None:
        return None

    @pytest.fixture
    def fp_df(
        self,
        amp_bcorr: FloatArray,
        time: FloatArray,
        prom: float,
        mpm: MapPeaksMixin,
        wlen: None,
    ) -> DataFrame[FindPeaks]:
        fp = mpm._set_fp_df(
            amp_bcorr,
            time,
            prom,
            wlen,
        )

        return fp

    @pytest.fixture
    def whh_rel_height(
        self,
    ) -> float:
        return 0.5

    @pytest.fixture
    def whh(
        self,
        mpm: MapPeaksMixin,
        amp_bcorr: FloatArray,
        fp_df: IntArray,
        whh_rel_height: float,
    ) -> DataFrame[WHH]:
        whh = mpm._set_whh_df(
            amp_bcorr,
            fp_df,
            whh_rel_height,
        )

        return whh

    @pytest.fixture
    def pb(
        self,
        mpm: MapPeaksMixin,
        amp_bcorr: FloatArray,
        fp_df: IntArray,
    ):
        pb = mpm._set_pb_df(
            amp_bcorr,
            fp_df,
        )
        return pb

    @pytest.fixture
    def pm(
        self,
        mpm: MapPeaksMixin,
        fp_df: DataFrame[FindPeaks],
        whh: DataFrame[WHH],
        pb: DataFrame[PeakBases],
    ) -> DataFrame[PeakMap]:
        pm = mpm._set_peak_map(
            fp_df,
            whh,
            pb,
        )
        return pm

    @pytest.fixture
    def mpp(
        self,
    ):
        mpp = MapPeakPlots()
        return mpp

    @pytest.fixture
    def peak_amp_colname(self) -> Literal["amp"]:
        return "amp"


class TestMapPeaksMixin(TestMapPeaksFix):
    """
    Before you get too carried away. To test you need to test the individual units - which should not have 'self' at the base level, and test the top level as well. Frankly the low level should be in their own classes without stored data, and only the top level should have stored data - the shinanigans you've been pulling for the last 12 hours may have been fun but ultimately they are bad practice, and a waste of time. Now to undo everything..

    2 classes - top level and low level.

    top level contains state storage, low level contains methods. MapPeaks, MapPeaksMixin, MapPeaksPlot Mixin, DataMixin


    """
    def test_set_fp_df(
        self,
        fp_df: DataFrame[FindPeaks],
    ):
        FindPeaks(fp_df)

    def test_set_whh(
        self,
        whh: DataFrame[WHH],
    ) -> None:
        WHH(whh)
        
    def test_set_pb(
        self,
        pb: DataFrame[PeakBases],
    ) -> None:
        PeakBases(pb)
        
    def test_set_peak_map(
        self,
        pm: DataFrame[PeakMap],
    ) -> None:
        PeakMap(pm)
        
class TestMapPeaks(TestMapPeaksFix):
    def test_map_peaks(
        self,
        mp: MapPeaks,
        amp_bcorr: FloatArray,
        time: FloatArray,
    )->None:
        
        mp = mp.map_peaks(amp_bcorr, time)
        
        
class TestMapPeakPlots(TestMapPeaksFix):
    def test_signal_plot(
        self,
        mpp: MapPeakPlots,
        bcorred_signal_df: DataFrame[SignalDFBCorr],
        time_colname: Literal["time"],
        bcorr_colname: Literal["amp_corrected"],
        mp,
    ) -> None:
        mpp._plot_signal_factory(bcorred_signal_df, time_colname, bcorr_colname)
        plt.show()

    def test_plot_peaks(
        self,
        mpp: MapPeakPlots,
        mpm: MapPeaksMixin,
        bcorred_signal_df: DataFrame[SignalDFBCorr],
        pm: DataFrame[PeakMap],
        time_colname: Literal["time"],
        bcorr_colname: Literal["amp_corrected"],
    ) -> None:
        
        PeakMap(pm)
        
        mpp._plot_signal_factory(bcorred_signal_df, time_colname, bcorr_colname)

        mpp._plot_peaks(
            pm,
            mpm._ptime_col,
            mpm._pmaxima_col,
        )

        plt.show()

    def test_plot_whh(
        self,
        mpp: MapPeakPlots,
        mpm: MapPeaksMixin,
        pm: DataFrame[PeakMap],
        bcorred_signal_df: DataFrame[SignalDFBCorr],
        time_colname: str,
        bcorr_colname: str,
        timestep: float,
    ) -> None:
        mpp._plot_signal_factory(bcorred_signal_df, time_colname, bcorr_colname)
        
        mpp.plot_whh(pm, mpm._whh_h_col, mpm._whh_l_col, mpm._whh_r_col, timestep)

        plt.show()

    def test_pb_plot(
        self,
        bcorred_signal_df: DataFrame[SignalDFBCorr],
        bcorr_colname: str,
        time_colname: str,
        mpp: MapPeakPlots,
        mpm: MapPeaksMixin,
        pm: DataFrame[PeakMap],
        timestep: float,
    ) -> None:
        mpp._plot_signal_factory(
            bcorred_signal_df,
            time_colname,
            bcorr_colname,
        )
        mpp._plot_peaks(
            pm,
            mpm._ptime_col,
            mpm._pmaxima_col,
        )
        mpp.plot_whh(pm, mpm._whh_h_col, mpm._whh_l_col, mpm._whh_r_col, timestep)
        
        mpp.plot_bases(
            pm,
            mpm._pb_h_col,
            mpm._pb_l_col,
            mpm._pb_r_col,
            timestep,
            plot_kwargs={"marker":"+"}
        )

        plt.show()

from hplc_py.map_signals.map_signals import MapWindowsMixin
        
class TestMapWindows:
    
    @pytest.fixture
    def mwm(
        self,
    )-> MapWindowsMixin:
        
        mwm = MapWindowsMixin()
        return mwm
    
    @pytest.fixture
    def mw(
        self,
    )->MapWindows:
        mw = MapWindows()
        
        return mw
    @pytest.fixture
    def pm(
        self,
        amp_bcorr: FloatArray,
        time: FloatArray,
    )-> DataFrame[PeakMap]:
        mp = MapPeaks()
        pm = mp.map_peaks(amp_bcorr, time,)
        return pm
    
    @pytest.fixture
    def time_idx(
        self,
        time: FloatArray,
    ):
        time_idx = np.arange(0, len(time), 1)
        
        return time_idx

    @pytest.fixture
    def left_bases(
        self,
        pm: DataFrame[PeakMap],
    )-> Series[pd.Int64Dtype]:
        left_bases: Series[pd.Int64Dtype] = pm[PeakMap.pb_left].astype(pd.Int64Dtype())
        return left_bases
    
    @pytest.fixture
    def right_bases(
        self,
        pm: DataFrame[PeakMap],
    )-> Series[pd.Int64Dtype]:
        right_bases: Series[pd.Int64Dtype] = pm[PeakMap.pb_right].astype(pd.Int64Dtype())
        return right_bases
    
    @pytest.fixture
    def intvls(
        self,
        mwm: MapWindowsMixin,
        left_bases: Series[pd.Int64Dtype],
        right_bases: Series[pd.Int64Dtype],
    )->Series[pd.Interval]:
        intvls: Series[pd.Interval] = mwm._interval_factory(left_bases, right_bases)
        return intvls
    
    @pytest.fixture
    def w_idxs(
        self,
        mwm: MapWindowsMixin,
        intvls: Series[pd.Interval],
    )->dict[int, list[int]]:
        w_idxs: dict[int, list[int]] = mwm._label_windows(intvls)
        return w_idxs
    
    @pytest.fixture
    def w_intvls(
        self,
        mwm: MapWindowsMixin,
        intvls: Series[pd.Interval],
        w_idxs: dict[int, list[int]],
    )->dict[int, pd.Interval]:
        w_intvls: dict[int, pd.Interval] = mwm._combine_intvls(intvls, w_idxs)
        return w_intvls

    @pytest.fixture
    def peak_wdws(
        self,
        mwm: MapWindowsMixin,
        time: FloatArray,
        w_intvls: dict[int, pd.Interval],
    )-> DataFrame[PeakWindows]:
        peak_windows = mwm._set_peak_windows(
        w_intvls,
        time,
        )
        
        return peak_windows
    
    @pytest.fixture
    def pwdwd_time(
        self,
        mwm: MapWindowsMixin,
        time: FloatArray,
        peak_wdws: DataFrame[PeakWindows],
    )->DataFrame[PWdwdTime]:
        
        pwdwd_time: DataFrame[PWdwdTime] = mwm._set_peak_wndwd_time(
            time,
            peak_wdws,
        )
        
        return pwdwd_time
    
    @pytest.fixture
    def wdwd_time(
        self,
        mwm: MapWindowsMixin,
        pwdwd_time: DataFrame[PWdwdTime],
    )->DataFrame[WindowedTime]:
        
        wdwd_time: DataFrame[WindowedTime] = mwm._label_interpeaks(
            pwdwd_time, mwm.pwdt_sc.window_idx
        )
        
        return wdwd_time
    
    @pytest.fixture
    def wm(
        self,
        mw: MapWindows,
        bcorred_signal_df: DataFrame[OutSignalDF_Base],
        pm: DataFrame[PeakMap],
    )->DataFrame[WindowedSignalDF]:
        
        
        wdwd_sdf = mw.map_windows(pm['pb_left'], pm['pb_right'], bcorred_signal_df['time'].to_numpy(np.float64))
        
        return wdwd_sdf
    
    def test_interval_factory(
        self,
        intvls: Series[pd.Interval],
    )->None:
        
        if not isinstance(intvls, pd.Series):
            raise TypeError("expected pd.Series")
        elif intvls.empty:
            raise ValueError("intvls is empty")
        elif not intvls.index.dtype == np.int64:
            raise TypeError(f"Expected np.int64 index, got {intvls.index.dtype}")
        elif not intvls.index[0]==0:
            raise ValueError("Expected interval to start at 0")
        
        
    def test_label_windows(
        self,
        w_idxs: dict[int, list[int]],
    )->None:
        if not w_idxs:
            raise ValueError('w_idxs is empty')
        if not any(v for v in w_idxs.values()):
            raise ValueError('a w_idx is empty')
        if not isinstance(w_idxs, dict):
            raise TypeError(f"expected dict")
        elif not all(isinstance(l, list) for l in w_idxs.values()):
            raise TypeError('expected list')
        elif not all(isinstance(x, int) for v in w_idxs.values() for x in v):
            raise TypeError("expected values of window lists to be int")
    
    def test_combine_intvls(
        self,
        w_intvls: dict[str, pd.Interval],
        )->None:
        if not w_intvls:
            raise ValueError("w_intvls is empty")
        if not isinstance(w_intvls, dict):
            raise TypeError("expected dict")
        if not all(isinstance(intvl, pd.Interval) for intvl in w_intvls.values()):
            raise TypeError("expected pd.Interval")
        
    def test_set_peak_windows(
        self,
        peak_wdws: DataFrame[PeakWindows],
    ):
        PeakWindows(peak_wdws)
    
    def test_label_interpeaks(
        self,
        wdwd_time: DataFrame[WindowedTime],
    ):
        WindowedTime.validate(wdwd_time, lazy=True)
        
    def test_map_windows(
        self,
        wm: DataFrame[WindowedSignalDF],
    )->None:
        WindowedTime(wm, lazy=True)
        
    def test_join_wm_to_sdf(
        self,
        bcorred_signal_df: DataFrame[SignalDFBCorr],
        wm: DataFrame[WindowedTime],
    ):

        wsdf_ = pd.merge(bcorred_signal_df, wm, on='time_idx', how='left', validate="1:1", indicator=True)

        wsdf = wsdf_.drop(['time_y', '_merge'], axis=1).copy(deep=True)
        
        return wsdf
        
    
class TestWindowing:
    manypeakspath = "/Users/jonathan/hplc-py/tests/test_data/test_many_peaks.csv"
    asschrompath = "tests/test_data/test_assessment_chrom.csv"

    @pytest.fixture
    def all_ranges(
        self,
        chm: Chromatogram,
        norm_amp: FloatArray,
        peak_df: pt.DataFrame[PeakMap],
    ):
        ranges = chm._ms.compute_individual_peak_ranges(
            norm_amp,
            peak_df["rl_left"].to_numpy(np.float64),
            peak_df["rl_right"].to_numpy(np.float64),
        )

        for range in ranges:
            assert all(range)

        assert len(ranges) > 0
        return ranges

    @pytest.fixture
    def all_ranges_mask(self, chm: Chromatogram, all_ranges: list[IntArray]):
        mask = chm._ms.mask_subset_ranges(all_ranges)

        assert len(mask) > 0
        assert any(mask == True)

        return mask

    @pytest.fixture
    def ranges_with_subset(
        self,
        chm: Chromatogram,
        all_ranges_mask: npt.NDArray[np.bool_],
        all_ranges: list[IntArray],
    ):
        new_ranges = copy.deepcopy(all_ranges)

        # for the test data, there are no subset ranges, thus prior to setup all
        # values in the mask will be true

        new_ranges.append(new_ranges[-1])

        return new_ranges

    def test_norm_amp(self, norm_amp: FloatArray) -> None:
        assert len(norm_amp) > 0
        assert np.min(norm_amp) == 0
        assert np.max(norm_amp) == 1
        return None

    test_peak_df_kwargs = {
        "argnames": [
            # "datapath",
            "schema"
        ],
        "argvalues": [
            # (
            #     manypeakspath,
            #     OutPeakDF_ManyPeaks
            # ),
            (
                # asschrompath,
                OutPeakDFAssChrom,
            ),
        ],
    }

    @pytest.mark.parametrize(**test_peak_df_kwargs)
    def test_peak_df(
        self,
        peak_df: pt.DataFrame,
        bcorred_signal_df: pd.DataFrame,
        bcorr_colname: str,
        schema,
    ):
        plt.scatter(peak_df["time_idx"], peak_df["peak_amp"])
        plt.show()
        try:
            schema(peak_df)
        except Exception as e:
            print("")
            print(
                interpret_model(peak_df, "OutPeakDF_Base", is_base=True),
            )

            print(
                interpret_model(
                    peak_df,
                    "OutPeakDFAssChrom",
                    "OutPeakBase",
                    check_dict={col: "eq" for col in peak_df},
                ),
            )
            raise ValueError(e)

    test_window_df_kwargs = {
        "argnames": [
            # "datapath",
            "schema"
        ],
        "argvalues": [
            # (manypeakspath, OutWindowDF_ManyPeaks),
            (
                # asschrompath,
                OutWindowDF_AssChrom,
            ),
        ],
    }

    @pytest.mark.parametrize(**test_window_df_kwargs)
    def test_window_df(
        self,
        window_df: pt.DataFrame[OutWindowDF_Base],
        schema,
    ):
        try:
            schema(window_df)
        except Exception as e:
            print(
                interpret_model(
                    window_df,
                    "OutWindowDF_Base",
                    "",
                    is_base=True,
                )
            )
            print(
                interpret_model(
                    window_df,
                    "OutWindowDF_AssChrom",
                    "OutWindowDF_Base",
                )
            )
            raise ValueError(e)

        return None

    @pa.check_types
    def test_windows_plot(
        self,
        chm: Chromatogram,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[PeakMap],
        window_df: pt.DataFrame[OutWindowDF_Base],
        time_col: str = "time_idx",
        y_col: str = "amp_corrected",
    ) -> None:
        # signal overlay

        fig, ax = plt.subplots()
        chm._ms.display_windows(
            peak_df,
            signal_df,
            window_df,
            time_col,
            y_col,
            ax,
        )

        # plt.show()

    def test_join_signal_peak_tbls(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[PeakMap],
    ):
        """
        test_join_signal_peak_tbls demonstrate successful joins of the signal and peak tables based on the time idx.
        """

        # default left join. Can join without explicitly setting index
        # but then you have to manually remove the join columns which
        # are duplicated

        peak_signal_join = (
            peak_df.set_index("time_idx")
            .join(signal_df.set_index("time_idx"), sort=True, validate="1:1")
            .reset_index()
        )

        # expect length to be the same as peak df
        assert len(peak_signal_join) == len(peak_df)
        # expect no NA
        assert peak_signal_join.isna().sum().sum() == 0

        return None

    def test_window_signal_join(
        self,
        window_df: pt.DataFrame[OutWindowDF_Base],
        signal_df: pt.DataFrame[OutSignalDF_Base],
    ):
        window_signal_join = window_df.set_index("time_idx").join(
            signal_df.set_index("time_idx"), sort=True, validate="1:1"
        )

        # left join onto window_df, expect row counts to be the same
        assert len(window_signal_join) == len(window_df)

        # expect no NAs as every index should match.
        assert window_signal_join.isna().sum().sum() == 0

    def test_assign_windows_exec(
        self,
        chm: Chromatogram,
        time: FloatArray,
        timestep: float,
        amp_bcorr: FloatArray,
    ) -> None:
        if amp_bcorr.ndim != 1:
            raise ValueError

        assert len(time) > 0
        assert len(amp_bcorr) > 0

        try:
            peak_df, window_df = chm._ms.profile_peaks_assign_windows(
                time,
                amp_bcorr,
                timestep,
            )
        except Exception as e:
            raise RuntimeError(e)

    @pytest.fixture
    def window_df_main(
        self,
        acr: AssChromResults,
    ):
        df = acr.tables["window_df"]
        df = df.reindex(labels=["time_idx", "window_idx", "window_type"], axis=1)

        # assign sw_idx as 1
        sw_df = (
            df.groupby(["window_type", "window_idx"])["time_idx"]
            .agg("first")
            .sort_values()
            .to_frame()
            .assign(**{"sw_idx": 1})
            .assign(
                **{"sw_idx": lambda df: df["sw_idx"].cumsum().astype(pd.Int64Dtype())}
            )
            .reset_index()
            .set_index("time_idx")
            .loc[:, ["sw_idx"]]
        )

        df = (
            df.set_index("time_idx")
            .join(sw_df, how="left")
            .assign(**{"sw_idx": lambda df: df["sw_idx"].ffill()})
            .reset_index()
        )

        df = df.astype(
            {
                "window_type": pd.StringDtype(),
                "time_idx": pd.Int64Dtype(),
                "window_idx": pd.Int64Dtype(),
            }
        )

        return df

    # @pytest.mark.xf;ail
    def test_assign_windows_compare_main(
        self,
        chm: Chromatogram,
        window_df: pt.DataFrame[OutWindowDF_Base],
        window_df_main: pt.DataFrame[OutWindowDF_Base],
        peak_df: pt.DataFrame[PeakMap],
        signal_df: pt.DataFrame[OutSignalDF_Base],
    ):
        """ """

        try:
            assert_frame_equal(window_df, window_df_main)

        except Exception as e:
            fig, axs = plt.subplots(ncols=2)
            chm._ms.display_windows(peak_df, signal_df, window_df_main, ax=axs[0])
            chm._ms.display_windows(peak_df, signal_df, window_df, ax=axs[1])
            plt.show()

            diff = window_df_main["sw_idx"] - window_df["sw_idx"]

            plt.plot(window_df_main["sw_idx"], label="main")
            plt.plot(window_df["sw_idx"], label="me")
            plt.legend()
            plt.show()
            raise RuntimeError(e)

        return None


class TestDataPrepper:
    manypeakspath = "tests/test_data/test_many_peaks.csv"
    asschrompath = "tests/test_data/test_assessment_chrom.csv"

    """
    The overall logic of the deconvolution module is as follows:
    
    1. iterating through each peak window:
        1. Iterating through each peak:
            1. build initial guesses as:
                - amplitude: the peak maxima,
                - location: peak time_idx,
                - width: peak width,
                - skew: 0
            2. build default bounds as:
                - amplitude: 10% peak maxima, 1000% * peak maxima.
                - location: peak window time min, peak window time max
                - width: the timestep, half the width of the window
                - skew: between negative and positive infinity
            3. add custom bounds
            4. add peak specific bounds
        5. submit extracted values to `curve_fit`
        ...
    
    so we could construct new tables which consist of the initial guesses, upper bounds and lower bounds for each peak in each window, i.e.:
    
    | # |     table_name   | window | peak | amplitude | location | width | skew |
    | 0 |  initial guesses |   1    |   1  |     70    |    200   |   10  |   0  |
    
    | # | table_name | window | peak | bound |  amplitude | location | width | skew |
    | 0 |    bounds  |    1   |   1  |   lb  |      7     |    100   | 0.009 | -inf |
    | 1 |    bounds  |    1   |   1  |   ub  |     700    |    300   |  100  | +inf |
    
    and go from there.
    
    The initial guess table needs the peak idx to be labelled with windows. since they both ahve the time index, thats fine. we also need the amplitudes from signal df.
    
    2023-12-08 10:16:41
    
    This test class now contains methods pertaining to the preparation stage of the deconvolution process.
    """

    def test_find_integration_range(
        self,
        dp: DataPrepper,
        bcorrected_signal_df: pt.DataFrame[OutSignalDF_Base],
    ) -> None:
        """
        find the integration range in time units for the given user input: note: note sure
        how to ttest this currently..

        TODO: add better test
        """

        tr = dp.find_integration_range(
            bcorrected_signal_df["time_idx"],  # type: ignore
            [30, 40],
        )

        assert pd.Series(tr).isin(bcorrected_signal_df["time_idx"]).all()

    @pa.check_input  # type: ignore
    @pytest.mark.parametrize(
        [
            # "datapath",
            "schema"
        ],
        [
            (
                # asschrompath,
                OutInitialGuessAssChrom,
            ),
        ],
    )
    def test_p0_factory_exec(
        self,
        p0_df: pt.DataFrame[OutInitialGuessBase],
        schema: pa.DataFrameSchema,
    ):
        """
        Test the initial guess factory output against the dataset-specific schema.
        """

        try:
            schema(p0_df)
        except Exception as e:
            # schema_error_str(schema, e, p0_df)
            print("")

            print(
                interpret_model(
                    p0_df,
                    "OutInitialGuessBase",
                    is_base=True,
                )
            )

            print("")

            schema_str = interpret_model(
                p0_df,
                "OutInitialGuessAssChrom",
                "OutInitialGuessBase",
            )

            print(schema_str)
            raise ValueError(e)

    def test_get_loc_bounds(
        self,
        chm: Chromatogram,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[PeakMap],
        window_df: pt.DataFrame[OutWindowDF_Base],
    ):
        class LocBounds(pa.DataFrameModel):
            window_idx: pd.Int64Dtype = pa.Field(eq=[1, 1, 1, 2])
            peak_idx: pd.Int64Dtype = pa.Field(eq=[0, 1, 2, 3])
            param: str = pa.Field(eq=["loc"] * 4)
            lb: pd.Float64Dtype = pa.Field(in_range={"min_value": 0, "max_value": 150})
            ub: pd.Float64Dtype = pa.Field(in_range={"min_value": 0, "max_value": 150})

        loc_bounds = chm._deconvolve.dataprepper.get_loc_bounds(
            signal_df, peak_df, window_df
        )

        LocBounds(loc_bounds)
        # try:
        # except Exception as e:
        #     assert False, str(e)+"\n"+str(loc_bounds)

        # assert isinstance(loc_bounds, pd.DataFrame)

    @pa.check_types
    @pytest.mark.parametrize(
        [
            # "datapath",
            "schema"
        ],
        [
            (
                # asschrompath,
                OutDefaultBoundsAssChrom,
            )
        ],
    )
    def test_default_bounds_factory(
        self,
        default_bounds: pt.DataFrame[OutDefaultBoundsBase],
        schema,
    ) -> None:
        """
        Define default bounds schemas
        """

        schema_tests(
            OutDefaultBoundsBase,
            schema,
            {"schema_name": "OutDefaultBoundsBase", "is_base": True},
            {
                "schema_name": "OutDefaultBoundsAssChrom",
                "check_dict": {col: "eq" for col in default_bounds.columns},
            },
            default_bounds,
        )

        return None


"""
2023-12-08 10:08:47

Since the trivial inputs work, we need to unit test p optimizer to expose the failcase data.
"""


@pytest.mark.xfail
class TestingCurveFit:
    @pytest.fixture
    def y(self, chm: Chromatogram, params, x):
        """
        Need:

        - [ ] time axis
        - [ ] params.
        """

        results = chm._deconvolve._fit_skewnorms(x, *params)

        return results

    def test_fit_skewnorms(self, y: NDArray[floating[Any]] | Literal[0]) -> None:
        """
        simply test if y is able to execute successfully
        """

        try:
            assert all(y)
        except Exception as e:
            raise RuntimeError(e)

    def test_curve_fit(
        self, chm: Chromatogram, params, x, y: NDArray[floating[Any]] | Literal[0]
    ):
        """
        test if optimize.curve_fit operates as expected
        """
        from scipy import optimize

        func = chm._deconvolve._fit_skewnorms

        try:
            popt, _ = optimize.curve_fit(func, x, y, params)
        except Exception as e:
            raise RuntimeError(e)

        popt = popt.reshape(2, 4)

        window_dict = {}
        for peak_idx, p in enumerate(popt):
            window_dict[f"peak_{peak_idx + 1}"] = {
                "amplitude": p[0],
                "retention_time": p[1],
                "scale": p[2],
                "alpha": p[3],
                "area": chm._deconvolve._compute_skewnorm(x, *p).sum(),
                "reconstructed_signal": chm._deconvolve._compute_skewnorm(x, *p),
            }


def test_popt_to_parquet(
    popt_df: Any,
    popt_parqpath: Literal["/Users/jonathan/hplc-py/tests/jonathan_tests/assch…"],
):
    """
    A function used to produce a parquet file of a popt df. I suppose it itself acts as a test, and means that whenever i run the full suite the file will be refreshed.
    """

    popt_df.to_parquet(popt_parqpath)


class TestDeconvolver:
    manypeakspath = "tests/test_data/test_many_peaks.csv"
    asschrompath = "tests/test_data/test_assessment_chrom.csv"

    @pytest.mark.parametrize(
        [
            # "datapath",
            "schema"
        ],
        [
            # (
            #     manypeakspath,
            #     OutWindowedSignalManyPeaks,
            # ),
            (
                # asschrompath,
                OutWindowedSignalAssChrom,
            ),
        ],
    )
    def test_windowed_signal_df(
        self,
        windowed_signal_df: OutWindowedSignalBase,
        schema,
    ) -> None:
        schema_tests(
            OutWindowedSignalBase,
            OutWindowedSignalAssChrom,
            {"schema_name": "OutWindowedSignalBase", "is_base": True},
            {
                "schema_name": "OutWindowedSignalAssChrom",
                "inherit_from": "OutWindowedSignalBase",
            },
            windowed_signal_df,
        )

    @pa.check_types
    @pytest.mark.parametrize(
        [
            # "datapath",
            "schema"
        ],
        [
            # (manypeakspath,OutParamManyPeaks,),
            (
                # asschrompath,
                OutParamsAssChrom,
            ),
        ],
    )
    def test_param_df_factory(
        self,
        my_param_df: pt.DataFrame[OutParamsBase],
        schema,
    ) -> None:
        schema_tests(
            OutParamsBase,
            schema,
            {"schema_name": "OutParamsBase", "is_base": True},
            {
                "schema_name": "OutParamsAssChrom",
                "inherit_from": "OutParamsBase",
                "check_dict": {col: "eq" for col in my_param_df},
            },
            my_param_df,
        )

        return None

    @pa.check_types
    @pytest.fixture
    def curve_fit_params(
        self,
        chm: Chromatogram,
        window: int,
        windowed_signal_df: pt.DataFrame[OutWindowedSignalBase],
        my_param_df: pt.DataFrame[OutParamsBase],
    ):
        params = chm._deconvolve._prep_for_curve_fit(
            window,
            windowed_signal_df,
            "amp_corrected",
            my_param_df,
        )
        return params

    @pa.check_types
    @pytest.mark.parametrize(
        [
            # "datapath",
            "window"
        ],
        [
            # (
            #     manypeakspath,
            #     1,
            # ),
            (
                # asschrompath,
                1,
            ),
            (
                # asschrompath,
                2,
            ),
        ],
    )
    def test_prep_for_curve_fit(
        self,
        curve_fit_params: tuple[
            Series[float], Series[float], Series[float], Series[float], Series[float]
        ],
    ):
        """
        TODO:
        - [ ] devise more explicit test.
        """
        results = curve_fit_params

        return None

    @pa.check_types
    @pytest.mark.parametrize(
        [
            # "datapath",
            "dset_schema"
        ],
        [
            (
                # asschrompath,
                OutPoptDF_AssChrom,
            )
        ],
    )
    def test_popt_factory(
        self,
        popt_df: pt.DataFrame,
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
            OutPoptDF_Base,
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
    # @pytest.mark.xfail

    @pytest.fixture
    def main_param_df(
        self,
        acr: AssChromResults,
    ):
        main_param_df = (
            acr.tables["adapted_param_tbl"].pipe(
                lambda df: df.astype(
                    {col: pd.Float64Dtype() for col in df if df[col].dtype == "float64"}
                ).astype(
                    {col: pd.Int64Dtype() for col in df if df[col].dtype == "int64"}
                )
            )
            #  .set_index(['window_idx','peak_idx','param'])
        )

        return main_param_df

    @pa.check_types
    def test_popt_factory_main_params_vs_my_params(
        self,
        my_param_df: DataFrame[OutParamsBase],
        main_param_df: DataFrame[OutParamsBase],
    ):
        assert_frame_equal(my_param_df, main_param_df)

        return None

    @pytest.fixture
    def main_chm_cls(
        self,
        in_signal: DataFrame,
    ):
        main_chm = hplc.quant.Chromatogram(in_signal)

        return main_chm

    def test_main_chm_exec(self, main_chm_cls: hplc.quant.Chromatogram):
        pass

    @pytest.fixture
    def main_gen_unmixed(
        self,
        stored_popt: pd.DataFrame,
        time: FloatArray,
        main_chm_cls: hplc.quant.Chromatogram,
    ):
        # windowed time df

        # iterate over popt generating a skewnorm dist for each peak, using time as x

        def gen_skewnorms(df: DataFrame, time: FloatArray):
            params = df[["amp", "loc", "whh", "skew"]].to_numpy()[0]

            skewnorm = main_chm_cls._compute_skewnorm(time, *params)

            idx = pd.Index(time)
            idx.name = "time"

            s = pd.Series(skewnorm, index=idx)

            return s

        main_gen_unmixed: pd.DataFrame = (
            stored_popt.groupby(["peak_idx"])
            .apply(gen_skewnorms, time)
            .stack()
            .to_frame("unmixed_amp")
            .reset_index()
        )

        return main_gen_unmixed

    def test_main_fit_skewnorms_against_mine(
        self,
        unmixed_df: DataFrame,
        main_gen_unmixed: DataFrame,
    ):
        unmixed_df = unmixed_df.drop(["time_idx"], axis=1)

        assert_frame_equal(unmixed_df, main_gen_unmixed)

    @pytest.mark.parametrize(
        [
            # 'datapath',
            "schema"
        ],
        [
            (
                # asschrompath,
                OutReconDF_AssChrom,
            )
        ],
    )
    def test_reconstruct_peak_signal(
        self, unmixed_df: pt.DataFrame[OutReconDFBase], schema
    ) -> None:
        base_schema_kwargs = {"schema_name": "OutReconDFBase", "is_base": True}
        dset_schema_kwargs = {
            "schema_name": "OutReconDF_AssChrom",
            "inherit_from": "OutReconDF_Base",
        }

        schema_tests(
            OutReconDFBase,
            schema,
            base_schema_kwargs,
            dset_schema_kwargs,
            unmixed_df,
        )

        return None

    @pytest.fixture
    def main_signal_df(
        self,
        acr: AssChromResults,
    ):
        print("")
        # ms_df = pd.DataFrame()
        # print(acr.tables.keys())
        ms_df = acr.tables["mixed_signals"]
        ms_df = ms_df.rename(
            {
                "x": "time",
                "y": "amp_raw",
                "y_corrected": "amp_corrected",
            },
            axis=1,
        )
        return ms_df

    def test_main_signal_df_exec(
        self,
        main_signal_df,
    ):
        print(main_signal_df)

    @pytest.fixture
    def main_unmixed_df(
        self,
        acr: AssChromResults,
        main_signal_df,
    ):
        unmixed_df = acr.tables["unmixed"].reset_index("time_idx")

        unmixed_df = unmixed_df.join(main_signal_df.loc[:, ["time"]])

        unmixed_df = unmixed_df.melt(
            id_vars=["time_idx", "time"], var_name="peak_idx", value_name="amp_unmixed"
        )
        unmixed_df = unmixed_df.loc[:, ["peak_idx", "time_idx", "time", "amp_unmixed"]]
        return unmixed_df

    def test_main_unmixed_df_exec(
        self,
        main_unmixed_df: Any,
    ):
        pass

    @pytest.fixture
    def main_window_df(
        self,
        acr: AssChromResults,
    ):
        w_df = acr.tables["window_df"]
        return w_df

    def test_main_window_df_exec(
        self,
        main_window_df,
    ):
        print("")
        print(main_window_df)

    def test_reconstruct_peak_signal_compare_main(
        self,
        unmixed_df: pt.DataFrame,
        main_unmixed_df: pt.DataFrame,
    ):
        idx = ["peak_idx", "time_idx", "time"]

        unmixed_df = unmixed_df.set_index(idx).rename(
            {"unmixed_amp": "amp_unmixed"}, axis=1
        )
        main_unmixed_df = main_unmixed_df.set_index(idx)

        print("")

        #   print(unmixed_df)
        #   print(main_unmixed_df)

        lsfx = "_mine"
        rsfx = "_main"
        compare_df = unmixed_df.join(
            main_unmixed_df, lsuffix=lsfx, rsuffix=rsfx, validate="1:1"
        )

        compare_df["diff"] = (
            compare_df["amp_unmixed" + lsfx] - compare_df["amp_unmixed" + rsfx]
        )
        compare_df["is_diff"] = compare_df["diff"] != 0

        print(compare_df)

        print(compare_df.loc[:, ["diff"]].agg(["min", "max"]))

    def test_peak_report(
        self,
        peak_report: pt.DataFrame[OutPeakReportBase],
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

    @pa.check_types
    def test_deconvolve_peaks(
        self,
        chm: Chromatogram,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[PeakMap],
        window_df: pt.DataFrame[OutWindowDF_Base],
        timestep: np.float64,
    ):
        popt_df, reconstructed_signals = chm._deconvolve.deconvolve_peaks(
            signal_df, peak_df, window_df, timestep
        )


class TestFitPeaks:
    """
    test the `fit_peaks` call, which performs the overall process to unmix the peaks and provide a peak table
    """

    @pytest.fixture
    def fit_peaks(
        self,
        chm: Chromatogram,
        in_signal: pt.DataFrame[SignalDFInBase],
    ):
        chm.set_signal_df(in_signal)

        popt_df, unmixed_df = chm.fit_peaks()

        return popt_df, unmixed_df

    @pytest.fixture
    def popt_df(self, fit_peaks: tuple[Any, Any]):
        return fit_peaks[0]

    @pytest.fixture
    def unmixed_df(self, fit_peaks: tuple[Any, Any]):
        return fit_peaks[1]

    @pa.check_types
    def test_fit_peaks_exec(
        self,
        fitted_chm: Chromatogram,
    ):
        assert fitted_chm

    def test_compare_timesteps(
        self,
        acr: AssChromResults,
        timestep: float,
    ):
        assert (
            acr.timestep == timestep
        ), f"timesteps are not equal. {acr.timestep, timestep}"


pytest.mark.skip


class TestShow:
    """
    Test the Show class methods
    """

    @pytest.fixture
    def fig_ax(self):
        return plt.subplots(1)

    @pytest.fixture
    def decon_out(
        self,
        chm: Chromatogram,
        signal_df: DataFrame,
        peak_df: DataFrame[PeakMap],
        window_df: DataFrame[Any],
        timestep: float,
    ):
        return chm._deconvolve.deconvolve_peaks(signal_df, peak_df, window_df, timestep)

    @pytest.fixture
    def popt_df(
        self, decon_out: tuple[DataFrame[OutPoptDF_Base], DataFrame[OutReconDFBase]]
    ):
        return decon_out[0]

    @pytest.fixture
    def popt_df(
        self, decon_out: tuple[DataFrame[OutPoptDF_Base], DataFrame[OutReconDFBase]]
    ):
        return decon_out[0]

    def test_plot_raw_chromatogram(
        self,
        fig_ax: tuple[Figure, Any],
        chm: Chromatogram,
        signal_df: OutSignalDF_Base,
    ):
        chm._show.plot_raw_chromatogram(
            signal_df,
            fig_ax[1],
        )

    def test_plot_reconstructed_signal(
        self,
        chm: Chromatogram,
        fig_ax: tuple[Figure, Any],
        unmixed_df: DataFrame[OutReconDFBase],
    ):
        chm._show.plot_reconstructed_signal(unmixed_df, fig_ax[1])

    def test_plot_individual_peaks(
        self,
        chm: Chromatogram,
        fig_ax: tuple[Figure, Any],
        unmixed_df: DataFrame[OutReconDFBase],
    ):
        ax = fig_ax[1]

        chm._show.plot_individual_peaks(
            unmixed_df,
            ax,
        )

    def test_plot_overlay(
        self,
        chm: Chromatogram,
        fig_ax: tuple[Figure, Any],
        signal_df: DataFrame,
        unmixed_df: DataFrame[OutReconDFBase],
    ):
        fig = fig_ax[0]
        ax = fig_ax[1]
        chm._show.plot_raw_chromatogram(
            signal_df,
            ax,
        )
        chm._show.plot_reconstructed_signal(
            unmixed_df,
            ax,
        )
        chm._show.plot_individual_peaks(
            unmixed_df,
            ax,
        )
