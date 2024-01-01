"""

"""

from pandas.core.frame import DataFrame
from pandera.typing.pandas import DataFrame
import pytest
import typing

import pandas as pd
import pandera as pa
import pandera.typing as pt

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt


import matplotlib as mpl

from scipy import integrate  # type: ignore


import copy


from hplc_py.quant import Chromatogram

from hplc_py.hplc_py_typing.hplc_py_typing import (
    SignalDFInBase,
    OutSignalDF_Base,
    OutPeakDF_Base,
    OutPeakDFAssChrom,
    OutWindowDF_Base,
    OutWindowDF_AssChrom,
    OutInitialGuessBase,
    OutInitialGuessAssChrom,
    OutDefaultBoundsBase,
    OutDefaultBoundsAssChrom,
    OutWindowedSignalBase,
    OutWindowedSignalAssChrom,
    OutParamsBase,
    OutParamAssChrom,
    OutPoptDF_Base,
    OutPoptDF_AssChrom,
    OutReconDFBase,
    OutReconDF_AssChrom,
    OutPeakReportBase,
    OutPeakReportAssChrom,
    interpret_model,
    schema_tests,
)

from hplc_py.quant import Chromatogram

pd.options.display.precision = 9

class TestInterpretModel:
    
    def schema_cls(
        self,
        schema_str,
    ):
        # instantiate the schema class
        exec( schema_str )
        
        # get the schema class object from locals
        schema_cls = locals()['InSampleDF']
        
        return schema_cls
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'col1':[1,2,3,4,5],
            'col2':['a','b','c','d','e'],
            'col3':[-1,-2,-3,-4,-5]
        })
    
    @pytest.fixture
    def test_gen_sample_df_schema(
        self,
        sample_df
    ):
        interpret_model(sample_df)
        
        return None
    
    @pytest.fixture
    def eq_schema_str(
        self,
        sample_df,
        ):

        check_dict = {col:'eq' for col in sample_df.columns} 
        schema_def_str = interpret_model(sample_df, 'InSampleDF', "", check_dict)
        
        return schema_def_str

    @pytest.fixture
    def isin_schema_str(
        self,
        sample_df,
        ):

        check_dict = {col:'isin' for col in sample_df.columns} 
        schema_def_str = interpret_model(sample_df, 'InSampleDF', "", check_dict)
        
        return schema_def_str
    
    @pytest.fixture
    def basic_stats_schema_str(
        self,
        sample_df
    ):
        numeric_cols=[]
        non_numeric_cols = []
        for col in sample_df:
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)
        
        # test numeric cols with 'basic_stats'
        numeric_col_check_dict = dict(zip(numeric_cols, ['basic_stats']*len(numeric_cols)))        
        non_numeric_col_check_dict = dict(zip(non_numeric_cols, ['isin']*len(non_numeric_cols)))
        
        check_dict = dict(**numeric_col_check_dict, **non_numeric_col_check_dict)
        schema_def_str = interpret_model(sample_df, "InSampleDF","", check_dict)
        
        return schema_def_str
    
    def test_eq_schema(
        self,
        sample_df,
        eq_schema_str,
                       ):
        '''
        Test whether the 'equals' schema works as expected
        '''
        schema = self.schema_cls(eq_schema_str)
        schema(sample_df)
    
    def test_isin_schema(
        self,
        sample_df,
        isin_schema_str,
                       ):
        '''
        Test whether the 'isin' schema works as expected
        '''
        schema = self.schema_cls(isin_schema_str)
        schema(sample_df)

    def test_basicstats_schema(
        self,
        sample_df,
        basic_stats_schema_str,
                       ):
        '''
        Test whether the 'basic_stats' schema works as expected
        '''
        
        schema = self.schema_cls(basic_stats_schema_str)
        schema(sample_df)
        
    def test_base_schema(
        self,
        sample_df
    ):
        
        # schema = self.schema_cls(sample_df)
        print(interpret_model(
            sample_df,
            'TestBaseSchema',
            is_base=True
        ))
        
        
    
        
    from hplc_py.hplc_py_typing.hplc_py_typing import SignalDFInAssChrom

    # @pa.check_types
    # def test_schema_eq(
    #     self,
    #     sample_df: pt.DataFrame[InSampleDF],
    # ):
    #     pass
        
            
    

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

def test_acr(acr):
    assert acr
    pass

def test_param_df_adapter(acr):
    param_df = acr.tables['asschrom_param_tbl']

    adapted_param_df = acr.adapt_param_df(param_df)
    try:
        OutParamsBase(adapted_param_df)
    except Exception as e:
        raise RuntimeError(e)

def check_df_exists(df):
    assert isinstance(df, pd.DataFrame)
    assert df.all
    return None

def test_target_window_df_exists(target_window_df):
    check_df_exists(target_window_df)
    pass

def test_get_asschrom_results(acr):
    """
    Simply test whether the member objects of AssChromResults are initialized.
    """
    
    for tbl in acr.tables:
        check_df_exists(acr.tables[tbl])




def test_timestep_exists_and_greater_than_zero(timestep):
    assert timestep
    assert timestep>0


def test_amp_raw_not_null(amp_raw):
    """
    for exploring shape and behavior of amp. a sandpit
    """
    assert all(amp_raw)


def test_bcorr_tuple_exists_and_len_2(bcorr_tuple):
    assert bcorr_tuple
    assert len(bcorr_tuple)==2
    pass

class TestLoadData:
        
    @pytest.fixture
    def valid_time_windows(self):
        return [[0, 5], [5, 15]]

    @pytest.fixture
    def invalid_time_window(self):
        return [[15, 5]]

    @pytest.fixture
    def test_timestep(self, timestep: float):
        assert timestep

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
    def debug_bcorr_df_main(
        self,
        acr
    ):
        """
        ['timestep', 'shift', 'n_iter', 'signal', 's_compressed', 's_compressed_prime', 'inv_tform', 'y_corrected', 'background']
        """
        return acr.tables['bcorr_dbug_tbl']
    
    @pytest.fixture
    def target_s_compressed_prime(
        self,
        debug_bcorr_df_main,
    ):
        return debug_bcorr_df_main['s_compressed_prime'].to_numpy(np.float64)
    
    @pytest.fixture
    def windowsize(
        self
    ):
        return 5
    
    @pytest.fixture
    def amp_clipped(
        self,
        chm: Chromatogram,
        amp_raw: npt.NDArray[np.float64],
    )-> npt.NDArray[np.float64]:
        return chm._baseline.shift_and_clip_amp(amp_raw)
    
    @pytest.fixture
    def s_compressed(
        self, chm: Chromatogram, amp_clipped: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # intensity raw compressed
        s_compressed = chm._baseline.compute_compressed_signal(amp_clipped)

        return s_compressed
    
    @pytest.fixture
    def s_compressed_main(
        self,
        acr,
    ):
        s_compressed = acr.tables['bcorr_dbug_tbl']['s_compressed']
        
        return s_compressed
    def test_s_compressed_against_main(
        self,
        s_compressed,
        s_compressed_main,
    ):
        '''
        test calculated s_compressed against the main version
        '''
        assert all(np.equal(s_compressed,s_compressed_main))
    
    def test_s_compressed_against_dbug_tbl(
        self,
        s_compressed,
        debug_bcorr_df,
    ):
        '''
        s_compressed also currently different to the debug tbl version.
        '''
        plt.plot(s_compressed, label='s_compressed')
        plt.plot(debug_bcorr_df.s_compressed, label='debug tbl')
        plt.legend()
        plt.show()
        
    def test_amp_raw_equals_main(
        self,
        amp_raw: npt.NDArray[np.float64],
        amp_raw_main: npt.NDArray[np.float64],
    ):
        assert np.all(np.equal(amp_raw,amp_raw_main))
    
    
    def test_amp_compressed_exists_and_is_array(
        self,
        s_compressed,
        ):
        
        assert np.all(s_compressed)
        assert isinstance(s_compressed, np.ndarray)
    
    @pytest.fixture
    def n_iter(
        self,
        chm: Chromatogram,
        windowsize,
        timestep,
    ):
        return chm._baseline.compute_n_iter(
            windowsize,
            timestep
        )
    
    
    @pytest.fixture
    def s_compressed_prime(
        self,
        chm: Chromatogram,
        s_compressed,
        n_iter,
    ):
      s_compressed_prime = chm._baseline.compute_s_compressed_minimum(
          s_compressed,
          n_iter,
      )
      return s_compressed_prime
  
    def test_amp_compressed_prime_against_main(
        self,
        s_compressed_prime: npt.NDArray[np.float64],
        target_s_compressed_prime: npt.NDArray[np.float64],
    ):
        
        if not np.all(s_compressed_prime==target_s_compressed_prime):
            raise ValueError("`amp_compressed_prime` does not equal target")
        return None
      
    def test_compute_inv_tfrom(
        self,
        chm: Chromatogram,
        amp_raw: npt.NDArray[np.float64],
    ) -> None:
        chm._baseline.compute_inv_tform(amp_raw)

    def test_correct_baseline(
        self,
        amp_raw: npt.NDArray[np.float64],
        chm: Chromatogram,
        time: npt.NDArray[np.float64],
        timestep,
    ) -> None:
        # pass the test if the area under the corrected signal is less than the area under the raw signal
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

        bcorr = chm._baseline.correct_baseline(
            added_baseline,
            timestep,
        )[0]

        baseline_auc = integrate.trapezoid(added_baseline, time)

        bcorr_auc = integrate.trapezoid(bcorr, time)

        assert baseline_auc > bcorr_auc
        
    @pytest.fixture
    def debug_bcorr_df(
        self,
        amp_raw: npt.NDArray[np.float64],
        chm: Chromatogram,
        time: npt.NDArray[np.float64],
        timestep,
    ):
        _, _ = chm._baseline.correct_baseline(
            amp_raw,
            timestep,
        )
        
        debug_df = chm._baseline._debug_bcorr_df
        return debug_df

    def test_debug_bcorr_df_compare_s_compressed_prime(
        self,
        s_compressed_prime,
        debug_bcorr_df
    ):
        '''
        I am expecting these two series to be identical, however they currently are not. the debug df is the same as the target.
        '''
        print()
        
        plt.plot(debug_bcorr_df["s_compressed_prime"], label='debug series')
        plt.plot(s_compressed_prime, label='isolated')
        plt.legend()
        plt.show()
        
    def test_debug_bcorr_df_compare_with_main(
        self,
        debug_bcorr_df,
        debug_bcorr_df_main,
    ):
        
        print("")
        print(debug_bcorr_df)
        print(debug_bcorr_df_main)
        
        for col in debug_bcorr_df:
            
            
            print(col)
            diff = debug_bcorr_df[col]-debug_bcorr_df_main[col]
            print((diff==0).all())
            print(diff.loc[(diff!=0)])
        
        # where are they not equal
        
        
        # plt.plot(debug_bcorr_df['s_compressed'], label='my compressed')
        # plt.plot(debug_bcorr_df_main['s_compressed'], label='main compressed')
        # plt.legend()
        # plt.show()

        # plt.plot(debug_bcorr_df['s_compressed_prime'], label='my prime')
        # plt.plot(debug_bcorr_df_main['s_compressed_prime'], label='main prime')
        
        plt.plot(debug_bcorr_df['y_corrected'], label='mine')
        plt.plot(debug_bcorr_df_main['y_corrected'], label='theirs')
        plt.legend()
        plt.show()
    
    def test_compare_timestep(
        self,
        timestep,
        debug_bcorr_df
    ):
        print("")
        print(timestep)
        print(debug_bcorr_df)
        print(debug_bcorr_df.dtypes)
        
        difference = timestep - debug_bcorr_df.iloc[0]
        
        print(difference)
        return None
        


class TestWindowing:
    manypeakspath = "/Users/jonathan/hplc-py/tests/test_data/test_many_peaks.csv"
    asschrompath = "tests/test_data/test_assessment_chrom.csv"

    @pytest.fixture
    def all_ranges(
        self,
        chm: Chromatogram,
        norm_amp: npt.NDArray[np.float64],
        peak_df: pt.DataFrame[OutPeakDF_Base],
    ):
        ranges = chm._findwindows.compute_individual_peak_ranges(
            norm_amp,
            peak_df["rl_left"].to_numpy(np.float64),
            peak_df["rl_right"].to_numpy(np.float64),
        )

        for range in ranges:
            assert all(range)

        assert len(ranges) > 0
        return ranges

    @pytest.fixture
    def all_ranges_mask(
        self, chm: Chromatogram, all_ranges: list[npt.NDArray[np.int64]]
    ):
        mask = chm._findwindows.mask_subset_ranges(all_ranges)

        assert len(mask) > 0
        assert any(mask == True)

        return mask

    @pytest.fixture
    def ranges_with_subset(
        self,
        chm: Chromatogram,
        all_ranges_mask: npt.NDArray[np.bool_],
        all_ranges: list[npt.NDArray[np.int64]],
    ):
        new_ranges = copy.deepcopy(all_ranges)

        # for the test data, there are no subset ranges, thus prior to setup all
        # values in the mask will be true

        new_ranges.append(new_ranges[-1])

        return new_ranges

    def test_norm_amp(self, norm_amp: npt.NDArray[np.float64]) -> None:
        assert len(norm_amp) > 0
        assert np.min(norm_amp) == 0
        assert np.max(norm_amp) == 1
        return None

    test_peak_df_kwargs = {
        "argnames": ["datapath", "schema"],
        "argvalues": [
            # (
            #     manypeakspath,
            #     OutPeakDF_ManyPeaks
            # ),
            (
                asschrompath,
                OutPeakDFAssChrom,
            ),
        ],
    }

    @pytest.mark.parametrize(**test_peak_df_kwargs)
    def test_peak_df(
        self,
        peak_df: pt.DataFrame,
        schema,
    ):
        print("")
        print(
            interpret_model(
                peak_df,
                'OutPeakDF_Base',
                is_base=True
                ),
                )
        try:
            schema(peak_df)
        except Exception as e:
            
            print(
                interpret_model(
                    peak_df,
                    'OutPeakDFAssChrom',
                    "OutPeakBase",
                    check_dict = {col: 'eq' for col in peak_df}
                    ),
                  )
            raise ValueError(e)

    def test_peak_df_viz(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        timestep,
    ):
        """
        overlay the plot metrics on the peaks
        """
        plt.style.use("ggplot")

        def pplot(peak_df, signal_df):
                
            # signal

            plt.plot(signal_df["time_idx"], signal_df["amp_corrected"], c="blue")

            # peaks

            plt.scatter(peak_df["time_idx"], peak_df["amp_corrected"], c="red")

            # # left and right bases of each width measurement
            # # the left and right values are values of the x axis, so simply plot them with the 'height' as the y value

            # # whh measurement

            plt.hlines(
                peak_df["whhh"],
                peak_df["whh_left"],
                peak_df["whh_right"],
                label="whh"
            )
            print("")

            # # 'relative height' width

            # plt.hlines(
            #     # peak_df['rl_wh'],
            #     [-0.1] * len(peak_df),
            #     peak_df["rl_left"],
            #     peak_df["rl_right"],
            #     label="rl width",
            #     color="green",
            # )
            plt.legend()
            plt.show()
            print(peak_df)

            return None

        peak_df = (
            peak_df.set_index("time_idx")
            .join(
                signal_df.set_index("time_idx"),
                how="left",
            )
            .reset_index()
            .set_index("peak_idx")
            .reset_index()
        )  # type: ignore

        pplot(peak_df, signal_df)

        print(signal_df.head())

        print(peak_df)

        print(timestep)

    test_window_df_kwargs = {
        "argnames": ["datapath", "schema"],
        "argvalues": [
            # (manypeakspath, OutWindowDF_ManyPeaks),
            (
                asschrompath,
                OutWindowDF_AssChrom,
            ),
        ],
    }

    @pytest.mark.parametrize(**test_window_df_kwargs)
    @pa.check_types
    def test_window_df(
        self,
        window_df: pt.DataFrame[OutWindowDF_Base],
        schema,
    ):  

        try:
            schema(window_df)
        except Exception as e:
            print(interpret_model(
                window_df,
                "OutWindowDF_Base",
                "",
                is_base=True,
            ))
        
            print(interpret_model(window_df,
                                "OutWindowDF_AssChrom",
                                "OutWindowDF_Base",
                                ))
            raise ValueError(e)

        return None

    test_windows_plot_kwargs = {
        "argnames": ["datapath"],
        "argvalues": [(manypeakspath,), (asschrompath,)],
    }

    @pytest.mark.parametrize(**test_windows_plot_kwargs)
    @pa.check_types
    def test_windows_plot(
        self,
        chm: Chromatogram,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
        time_col: str='time_idx',
        y_col: str='amp_corrected',
    ) -> None:
        # signal overlay
        
        chm._findwindows.display_windows(
            peak_df,
            signal_df,
            window_df,
            time_col,
            y_col,
        )

    def test_join_signal_peak_tbls(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
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

    def test_assign_windows(
        self,
        chm: Chromatogram,
        time: npt.NDArray[np.float64],
        timestep: float,
        amp_bcorr: npt.NDArray[np.float64],
    ) -> None:
        if amp_bcorr.ndim != 1:
            raise ValueError

        assert len(time) > 0
        assert len(amp_bcorr) > 0

        try:
            chm._findwindows.profile_peaks_assign_windows(
                time,
                amp_bcorr,
                timestep,
            )
        except Exception as e:
            raise RuntimeError(e)
    
    @pytest.mark.xfail
    @pa.check_types
    def test_assign_windows_compare_main(
        self,
        chm: Chromatogram,
        window_df: pt.DataFrame[OutWindowDF_Base],
        window_df_main: pt.DataFrame[OutWindowDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        signal_df: pt.DataFrame[OutSignalDF_Base],
    ):
        """
        As of 2023-12-19 16:25:54 the window dfs are numerically noncomparable due to fundamental differences in how a window is defined, i.e. i define a window as peak-specific, they define both interpeak and peak windows as windows. at the moment, visual comparison is acceptable.
        """
        print("")
        print(window_df)
        window_df_main = window_df_main.reindex(labels=['window_idx','time_idx','window_type'],axis=1)
        print(window_df_main)
        print(signal_df)
        
        fig, axs = plt.subplots(ncols=2)
        chm._findwindows.display_windows(
            peak_df,
            signal_df,
            window_df_main,
            ax=axs[0]
        )
        chm._findwindows.display_windows(
            peak_df,
            signal_df,
            window_df,
            ax=axs[1]
        )
        fig.show()
        plt.show()
        # print(window_df.compare(window_df_main))
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
        self, chm: Chromatogram, signal_df: pt.DataFrame[OutSignalDF_Base]
    ) -> None:
        """
        find the integration range in time units for the given user input: note: note sure
        how to ttest this currently..

        TODO: add better test
        """

        tr = chm._deconvolve.dataprepper.find_integration_range(
            signal_df["time_idx"],  # type: ignore
            [30, 40],
        )

        assert pd.Series(tr).isin(signal_df["time_idx"]).all()

    @pytest.mark.parametrize(
        ["datapath", "schema"],
        [
            # (
            #     manypeakspath,
            #     OutInitialGuessManyPeaks,
            # ),
            (
                asschrompath,
                OutInitialGuessAssChrom,
            ),
        ],
    )
    def test_p0_factory(
        self,
        p0_df: pt.DataFrame[OutInitialGuessBase],
        schema,
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
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
        ):
        
        class LocBounds(pa.DataFrameModel):
            window_idx: pd.Int64Dtype =pa.Field(eq=[1,1,1,2])
            peak_idx: pd.Int64Dtype =pa.Field(eq=[0,1,2,3])
            param: str =pa.Field(eq=['loc']*4)
            lb: pd.Float64Dtype = pa.Field(in_range={'min_value':0,'max_value':150})
            ub: pd.Float64Dtype = pa.Field(in_range={'min_value':0,'max_value':150})
            
            
        loc_bounds = chm._deconvolve.dataprepper.get_loc_bounds(signal_df,peak_df, window_df)
        
        LocBounds(loc_bounds)
        # try:
        # except Exception as e:
        #     assert False, str(e)+"\n"+str(loc_bounds)
        
        # assert isinstance(loc_bounds, pd.DataFrame)
        
    @pa.check_types
    @pytest.mark.parametrize(
        ["datapath", "schema"],
        [
            # (
            #     manypeakspath,
            #     OutDefaultBoundsManyPeaks,
            # ),
            (
                asschrompath,
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

        try:
            schema(default_bounds)
        except Exception as e:
            schema_str = interpret_model(
                "OutDefaultBoundsAssChrom",
                default_bounds,
                "OutDefaultBoundsBase",
            )
            print(schema_str)
            raise ValueError(e)
        return None

        return None


"""
2023-12-08 10:08:47

Since the trivial inputs work, we need to unit test p optimizer to expose the failcase data.
"""



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

    def test_fit_skewnorms(self, y) -> None:
        """
        simply test if y is able to execute successfully
        """

        try:
            assert all(y)
        except Exception as e:
            raise RuntimeError(e)

    def test_curve_fit(self, chm: Chromatogram, params, x, y):
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



def test_popt_to_parquet(popt_df, popt_parqpath):
    """
    A function used to produce a parquet file of a popt df. I suppose it itself acts as a test, and means that whenever i run the full suite the file will be refreshed.
    """

    popt_df.to_parquet(popt_parqpath)


class TestDeconvolver():
    manypeakspath = "tests/test_data/test_many_peaks.csv"
    asschrompath = "tests/test_data/test_assessment_chrom.csv"

    @pytest.mark.parametrize(
        ["datapath", "schema"],
        [
            # (
            #     manypeakspath,
            #     OutWindowedSignalManyPeaks,
            # ),
            (
                asschrompath,
                OutWindowedSignalAssChrom,
            ),
        ],
    )
    def test_windowed_signal_df(
        self,
        windowed_signal_df,
        schema,
    ) -> None:
        try:
            schema(windowed_signal_df)
        except Exception as e:
            schema_str = interpret_model(windowed_signal_df, 'OutWindowedSignalAssChrom', "OutWindowedSignalBase", check_dict={col:'basic_stats' for col in windowed_signal_df})
            print(schema_str)

    @pa.check_types
    @pytest.mark.parametrize(
        ["datapath", "schema"],
        [
            # (manypeakspath,OutParamManyPeaks,),
            (
                asschrompath,
                OutParamAssChrom,
            ),
        ],
    )
    def test_param_df_factory(
        self,
        param_df: pt.DataFrame[OutParamsBase],
        schema,
    ) -> None:
        
        try:
            schema(param_df)
        except Exception as e:
            
            schema_str = interpret_model(param_df, "OutParamAssChrom","OutParamsBase", {col:'eq' for col in param_df})
            print(schema_str)
                
            raise ValueError(e)

        return None

    @pa.check_types
    @pytest.fixture
    def curve_fit_params(
        self,
        chm: Chromatogram,
        window: int,
        windowed_signal_df: pt.DataFrame[OutWindowedSignalBase],
        param_df: pt.DataFrame[OutParamsBase],
    ):
        params = chm._deconvolve._prep_for_curve_fit(
            window,
            windowed_signal_df,
            'amp_corrected',
            param_df,
        )
        return params

    @pa.check_types
    @pytest.mark.parametrize(
        ["datapath", "window"],
        [
            # (
            #     manypeakspath,
            #     1,
            # ),
            (
                asschrompath,
                1,
            ),
            (
                asschrompath,
                2,
            ),
        ],
    )
    def test_prep_for_curve_fit(
        self,
        curve_fit_params,
    ):
        """
        TODO:
        - [ ] devise more explicit test.
        """
        results = curve_fit_params

        return None

    
    @pa.check_types
    @pytest.mark.parametrize(
        ["datapath", "schema"],
        [
            (
                asschrompath,
                OutPoptDF_AssChrom,
            )
        ],
    )
    def test_popt_factory(
        self,
        popt_df,
        schema,
    ):
        """
        TODO:
        - [ ] define dataset specific schemas
        - [ ] identify why algo needs more than 1200 iterations to minimize mine vs 33 for main
        - [ ] testing with the main adapted param_df, 24 iterations for the first window, 21 for the second. Whats the difference?
        
        Note: as of 2023-12-21 11:02:03 first window now takes 803 iterations. same window in main takes 70 iterations.
        """

        print("")
        print(
            interpret_model(
                popt_df,
                "OutPoptDF_Base",
                is_base=True,
            )
        )
        
        print("")
        print(
            interpret_model(
            popt_df,
            "OutPoptDF_AssChrom"
        )
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
    @pytest.mark.xfail
    def test_popt_factory_main_params_vs_my_params(
        self,
        chm: Chromatogram,
        acr,
        param_df,
        windowed_signal_df,
    ):
        # print("")
        # print(param_df)
        # print(param_df.columns)
        
        param_df: pd.DataFrame = (param_df
                    .set_index(['window_idx','peak_idx','param'])
                    )
        
        main_param_df = (acr
                         .tables['adapted_param_tbl']
                         .pipe(lambda df: 
                             df
                             .astype({col:pd.Float64Dtype() for col in df if df[col].dtype=='float64'})
                             .astype({col:pd.Int64Dtype() for col in df if df[col].dtype=='int64'})
                             )
                         .set_index(['window_idx','peak_idx','param'])
                         
                         )
        # test that all datatypes are equal
        print("")
        print(param_df.to_markdown())
        print("")
        print(main_param_df.to_markdown())
        # assert False
        
        if not param_df.dtypes.equals(main_param_df.dtypes):
            # print(param_df.dtypes)
            dtypes_neq_mask = ~(param_df.dtypes==main_param_df.dtypes)
            not_eq_param_df = param_df.dtypes[dtypes_neq_mask]
            not_eq_main_param_df = main_param_df.dtypes[dtypes_neq_mask]
            
            raise ValueError(f"dtypes are not equal. Mine:\n{not_eq_param_df}\nTheirs:\n{not_eq_main_param_df}")
        
        # check index dtypes samec
        if not param_df.index.dtypes.equals(main_param_df.index.dtypes):
            my_idx_dtypes = param_df.index.dtypes
            main_idx_dtypes = main_param_df.index.dtypes
            idx_dtype_neq_mask = ~(my_idx_dtypes==main_idx_dtypes)
            neq_param_df_idx_dtypes = my_idx_dtypes[idx_dtype_neq_mask]
            neq_main_param_df_idx_dtypes = main_idx_dtypes[idx_dtype_neq_mask]
            
            raise ValueError(f"index dtypes are not equal. Mine:\n{neq_param_df_idx_dtypes}\nTheirs:\n{neq_main_param_df_idx_dtypes}")
            
        if not param_df.index.equals(main_param_df.index):
            index_neq_mask = ~(param_df.index==main_param_df.index)
            neq_param_df_idx = param_df.index[index_neq_mask]
            neq_main_param_df_idx = main_param_df.index[index_neq_mask]
            
            raise ValueError(f"index values are not equal. Mine:\n{neq_param_df_idx}\nTheirs:\n{neq_main_param_df_idx}")

        if not param_df.columns.equals(main_param_df.columns):
            columns_neq_mask = ~(param_df.columns==main_param_df.columns)
            neq_param_df_cols = param_df.index[columns_neq_mask]
            neq_main_param_df_cols = main_param_df.index[columns_neq_mask]
            
            raise ValueError(f"columns are not equal. Mine:\n{neq_param_df_cols}\nTheirs:\n{neq_main_param_df_cols}")
        
        if not (isinstance(param_df, pd.DataFrame) & isinstance(main_param_df, pd.DataFrame)):
            raise ValueError(f"Expected DataFrames, got {type(param_df)}, {type(main_param_df)}")
        
        param_df = param_df.copy(deep=True)
        
        # for row, val in param_df.compare(main_param_df).items():
        #     print(row, val)
        
        compare_df = param_df.compare(main_param_df, result_names=('mine','main'))
        print("")
        for idx in compare_df.index:
            print(idx)
            print(compare_df.loc[[idx]])
            
        """
        So it appears that specific windows are causing the errors. this appears to me as though it is caused by my redefinition of the windows. Can I remember how they differ? no, but the summation in my memory is that I did not define interpeak windows, ergo the bound for each peak should be the same. evidently not.
        
        loc is defined as:
        ```
        'location': [v['time_range'].min(), v['time_range'].max()],
        ```
        whh is defined as:
        ```
        'scale': [self._dt, (v['time_range'].max() - v['time_range'].min())/2],
        ```
        
        In my code, whhh bounds are derived from the loc bounds, whose calculation mirrors the main:
        
        loc_bounds = (
            peak_df_window_df.loc[:, ["window_idx", "peak_idx"]]
            .set_index("window_idx")
            .join(
                pivot_window_df,
                how="left",
            )
            .reset_index()
        )
        
        which joins the peak indexes to the window bounds
        
        pretty sure I have an incomplete swap from time index to time units.
        """
        
        
        # for col in param_df.columns:
            
        #     s1 = param_df[col].to_numpy(np.float64)
        #     s2 = param_df[col].to_numpy(np.float64)
        #     not_eq = np.equal(s1, s2)
            
        #     print(s1[not_eq])
        #     print(s2[not_eq])
            
        #     raise ValueError("\n"+str(not_eq))
        
        # compare each column to account for diff dtype behavior
        diff_compars = {}
        for col in param_df.columns:
            try:
                is_diff_mask = param_df[col].ne(main_param_df[col])
            except Exception as e:
                raise ValueError(
                    str(e)+"\n\n"+str(param_df[col])+"\n\n"+str(main_param_df[col])
                    )
            
            if is_diff_mask.any():
                # side_by_side = pd.concat([param_df[col], main_param_df[col]], axis=1)
                # side_by_side = side_by_side[is_diff_mask]
                # raise ValueError("\n" + str(side_by_side))
                
                difference = param_df[col][is_diff_mask]-main_param_df[col][is_diff_mask]
                raise ValueError(
                    "\n"+str(difference)
                )
                
                
                raise ValueError(
                    "\n"+str(param_df[is_diff_mask])+"\n\n"+str(main_param_df[is_diff_mask])
                )
        
        # popt_df = chm._deconvolve._popt_factory(windowed_signal_df,
        #                               get_acr.adapted_param_df,
        #                               )
        # print(popt_df)
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
        self,
        unmixed_df: pt.DataFrame[OutReconDFBase],
        schema
    ) -> None:
        base_schema_kwargs = {"schema_name":"OutReconDFBase", "is_base":True}
        dset_schema_kwargs ={'schema_name':"OutReconDF_AssChrom", 'inherit_from':"OutReconDF_Base"}
    
        schema_tests(
            OutReconDFBase,
            schema,
            base_schema_kwargs,
            dset_schema_kwargs,
            unmixed_df,
        )
        
        return None
    
    
    def test_peak_report(
        self,
        peak_report: pt.DataFrame[OutPeakReportBase],
    ):
        
        
        schema_tests(
            OutPeakReportBase,
            OutPeakReportAssChrom,
            {'schema_name':'OutPeakReportBase',
             'is_base':True,
             },
            {'schema_name':'OutPeakReportAssChrom',
             'inherit_from':'OutPeakReportBase',                
            },
            peak_report,
            verbose=True,
                     )
        
        return None
            

    @pa.check_types
    @pytest.mark.parametrize(
        [
            "datapath",
            # 'schema'
        ],
        [
            (
                asschrompath,
                # OutPoptAssChrom,
            )
        ],
    )
    def test_deconvolve_peaks(
        self,
        chm: Chromatogram,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
        timestep: np.float64,
    ):
        popt_df, reconstructed_signals = chm._deconvolve.deconvolve_peaks(
            signal_df, peak_df, window_df, timestep
        )


class TestFitPeaks:
    """
    test the `fit_peaks` call, which performs the overall process to unmix the peaks and
    provide a peak table
    """

    @pytest.fixture
    def fit_peaks(
        self,
        chm: Chromatogram,
        in_signal: pt.DataFrame[SignalDFInBase],
    ):
        chm.load_data(in_signal)

        popt_df, unmixed_df = chm.fit_peaks()

        return popt_df, unmixed_df

    @pytest.fixture
    def popt_df(self, fit_peaks):
        return fit_peaks[0]

    @pytest.fixture
    def unmixed_df(self, fit_peaks):
        return fit_peaks[1]
    
    def test_fit_peaks(
        self,
        chm: Chromatogram,
        in_signal: pt.DataFrame[SignalDFInBase],
        amp_colname: str,
        time_colname: str,
        
    ):  
        chm.load_data(
            in_signal,
        )
        
        chm.fit_peaks(
            amp_colname,
            time_colname,
        )
        
        return None

    def test_compare_timesteps(
        self,
        acr,
        timestep,
    ):
        assert (
            acr.timestep == timestep
        ), f"timesteps are not equal. {acr.timestep, timestep}"

    def test_compare_param_dfs(self, param_df, acr):
        """
        method for comparing the adapted param df and my param df
        """

        # print(
        #         param_df.copy().compare(get_acr.adapted_param_df)
        # )
        print(f"\n{param_df}")
        print(f"\n{acr.adapted_param_df}")

        print(f"\n{param_df.dtypes}")
        print(f"\n{acr.adapted_param_df.dtypes}")

    def test_compare_opt_params(
        self,
        param_df,
        timestep,
        acr,
    ):
        # print(f"\n{popt_df}")

        print_main_df = (
            acr.param_df
            #  .query("param=='scale'")
        )
        print(f"\n{print_main_df}")

        num_cols = ["p0", "lb", "ub"]
        mask = (param_df["param"] == "loc") | (param_df["param"] == "whh")

        param_df.loc[mask, num_cols] = param_df.loc[mask, num_cols] * timestep

        print_my_df = (
            param_df
            #    .query("param=='whh'")
            #    .astype(float, errors='ignore')
        )
        print(f"\n{print_my_df}")

    """
    
    modifications to be made to popt:
    - [ ] express 'loc' as 'retention time' i.e. going back to time units
    - [ ] What is the main 'scale'
    - [ ] area needs to be added to the table
    - [ ] change order to loc, amp, scale, skew
    
    other mods:
    - [ ] convert parameter input to time units prior to curve fit to match behavior of main
    
    Notes: as it stands, my 'whh' is not equivalent to main 'scale'. Need to clarify that. Specifically, my initial guesses are 2x larger, my lower bounds are smaller by a factor of ten, but my upper bound is the same.
    
    The cause is that you are using the timestep for the lower bound, but the time_idx for the upper bound and guess, meaning that they are on different scales. Thus the transformation does not perform as intended, and results in the much smaller lb value than expected. The simplest solution I can see at this time is to convert the lb to 1 rather than the timestep, 1 being the smallest the peak can possibly be. This has the benefit of avoiding modification of code logic, and retention of time index scale rather than time unit scale.
    
    Note - the initial guess was off because I was missing the division by 2. the initial guess should be half the calculated width. gives the algo wiggle room i guess.
    
    Now i have the problem of an infinite iteration to solve.
    
    - [ ] achieve a successful optimization without infinite looping.
        - [ ] test popt_factory with the main params to see the performance
    """


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
        signal_df,
        peak_df,
        window_df,
        timestep,
    ):
        return chm._deconvolve.deconvolve_peaks(signal_df, peak_df, window_df, timestep)

    @pytest.fixture
    def popt_df(self, decon_out):
        return decon_out[0]

    @pytest.fixture
    def popt_df(self, decon_out):
        return decon_out[0]

    def test_plot_raw_chromatogram(
        self,
        fig_ax,
        chm: Chromatogram,
        signal_df: OutSignalDF_Base,
    ):
        chm.show.plot_raw_chromatogram(
            signal_df,
            fig_ax[1],
        )

        plt.legend()
        plt.show()

    def test_plot_reconstructed_signal(
        self,
        chm: Chromatogram,
        fig_ax,
        unmixed_df,
    ):
        chm.show.plot_reconstructed_signal(unmixed_df, fig_ax[1])
        plt.legend()
        plt.show()

    def test_plot_individual_peaks(
        self,
        chm: Chromatogram,
        fig_ax,
        unmixed_df,
    ):
        ax = fig_ax[1]

        chm.show.plot_individual_peaks(
            unmixed_df,
            ax,
        )

        plt.legend()
        plt.show()

    def test_plot_overlay(
        self,
        chm: Chromatogram,
        fig_ax,
        signal_df,
        unmixed_df,
    ):
        fig = fig_ax[0]
        ax = fig_ax[1]
        chm.show.plot_raw_chromatogram(
            signal_df,
            ax,
        )
        chm.show.plot_reconstructed_signal(
            unmixed_df,
            ax,
        )
        chm.show.plot_individual_peaks(
            unmixed_df,
            ax,
        )

        plt.legend()
        plt.show()

    