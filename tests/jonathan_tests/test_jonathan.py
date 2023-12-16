"""
TODO:

- [ ] write 'stored_popt' for both datasets- currently only testing on asschrom

TODO:

- [x] plot whh to understand what it is.
- [x] test asschrom dataset with original code to see what the width values should be
- [x] diagnose why my code produes a different value
    - reason 1 is that the peak widths are stored in time units rather than index units.
    - thats the only reason. same same, dont change a thing.
- [x] write up on the skew norm model to understand:
    - [x] why parameter choices are what they are
    - [ ] interpretability of the results, how reliable are they?
    - [ ] what are the danger points of the skewnorm model
        - the last two are currently unanswered.
- [x] adapt .show
    - [x] plot raw chromatogram
    - [x] plot inferred mixture (sum of reconstructed signals)
    - [x] plot mapped peaks (fill between)
    - [ ] add custom peaks subroutine
- [ ] identify why the fitted peaks do not match the original signal.
    - [x] define a fixture class that returns the parameters of the hplc_py calculation for the same dataset. Provide an interface that outputs the parameters in the same format as your definitions for easy comparison.
        - [x] output the results from the main env to a parquet file
        - [x] write the fixture to read the output
        - [x] write tests to compare my results with theirs.
        - [x] add a params_df output to the fixture, this being the lb, p0, ub in similar format to mine.
        - [x] add timestep to params_df
    - [x] write an adapter (proto decorator) to express the main calculated parameters as your format to feed to `_popt_factory` directly.
    - [ ] determine why your p0 values are rounded to 3 decimal places
    
        
- [ ] adapt map_peaks (?)
- [ ] adapt fit assessment module(s)
- [ ] add in the customization routines
- [ ] build the master method
- [ ] seperate initial peak profiling from window finding 
- [ ] make the modules more abstracted, expose intermediate values as class objects.


Notes:

"""

from typing import Literal
from pandas.core.arrays.base import ExtensionArray
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

from matplotlib.patches import Rectangle
import matplotlib as mpl

from scipy import integrate  # type: ignore


import copy


from hplc_py.quant import Chromatogram
from hplc_py.hplc_py_typing.hplc_py_typing import (
    SignalDFIn,
    OutSignalDF_Base,
    OutSignalDF_ManyPeaks,
    OutSignalDF_AssChrom,
    OutPeakDF_Base,
    OutPeakDF_ManyPeaks,
    OutPeakDF_AssChrom,
    OutWindowDF_Base,
    OutWindowDF_ManyPeaks,
    OutWindowDF_AssChrom,
    OutInitialGuessBase,
    OutInitialGuessManyPeaks,
    OutInitialGuessAssChrom,
    OutDefaultBoundsBase,
    OutDefaultBoundsManyPeaks,
    OutDefaultBoundsAssChrom,
    OutWindowedSignalBase,
    OutWindowedSignalManyPeaks,
    OutWindowedSignalAssChrom,
    OutParamsBase,
    OutParamAssChrom,
    OutParamManyPeaks,
    OutPoptBase,
    OutPoptManyPeaks,
    OutPoptAssChrom,
    OutReconDFBase,
    OutPeakReportBase,
    OutPeakReportAssChrom,
)

from hplc_py.hplc_py_typing.hplc_py_typing import (
    isArrayLike,
    OutPeakDF_Base,
    SignalDFIn,
    OutWindowDF_Base,
)
from hplc_py.quant import Chromatogram

import os

class TestSuperClass:
    '''
    simply contains the paths to the test files for test parametrization
    '''
    manypeakspath = "/Users/jonathan/hplc-py/tests/test_data/test_many_peaks.csv"
    asschrompath = "tests/test_data/test_assessment_chrom.csv"
    
class AssChromResults:
    '''
    The results of 'fit_peaks' on the 'test_assessment_chrom' dataset
    
    to add:
    
    - [x] timestep. Has been added as a column of 'param_df'
    
    '''
    def __init__(self):
        
        ptable_path = self.get_results_paths()['ptable_outpath']
        unmixed_path = self.get_results_paths()['unmixed_outpath']
        param_path = self.get_results_paths()['param_outpath']
        
        self.param_df = pd.read_parquet(param_path)
        self.adapted_param_df = self.adapt_param_df(self.param_df)
        self.ptable = pd.read_parquet(ptable_path)
        self.unmixed = pd.read_parquet(unmixed_path)
        self.timestep = self.param_df.loc[0,'timestep']
    
    def adapt_param_df(self,
                       param_df):
        '''
        adapt main param_df to be expressed as per the definition that `popt_factory` is expecting
        
        '''
        from scipy.optimize._lsq.common import in_bounds
        timestep = self.param_df.loc[0,'timestep']
        adapted_param_df = (
            param_df
            .assign(inbounds=lambda df: df.apply(
                lambda x: in_bounds(x["p0"], x["lb"], x["ub"]),
                axis=1,
            )
            )
            .assign(
                param=lambda df:
                    df['param'].replace({
                        'amplitude':'amp',
                        'location':'loc',
                        'scale':'whh',
                    })
            )
            .assign(
                param=lambda df: pd.Categorical(values=df['param'], categories=df['param'].unique(), ordered=True   
                )
            )
            )
        # divide by timestep to return to time index
        time_row_mask = adapted_param_df['param'].isin(['loc','whh'])
        num_cols  = ['p0','lb','ub']
        
        adapted_param_df.loc[time_row_mask, num_cols] = adapted_param_df.loc[time_row_mask, num_cols]/timestep
        
        return adapted_param_df
    
    def get_results_paths(self):
        out_basepath = "/Users/jonathan/hplc-py/tests/jonathan_tests/main_asschrom_results"

        ptable_outpath = os.path.join(out_basepath, "peak_table.parquet")
        unmixed_outpath = os.path.join(out_basepath, "unmixed.parquet")
        param_df_outpath = os.path.join(out_basepath, "param_df.parquet")
        
        return {
            'param_outpath': param_df_outpath,
            'ptable_outpath':ptable_outpath,
            'unmixed_outpath':unmixed_outpath,
        }
        
@pytest.fixture
def get_acr():
    acr = AssChromResults()
    return acr

def test_param_df_adapter(
    get_acr
):
    param_df = get_acr.param_df
    
    adapted_param_df = get_acr.adapt_param_df(param_df)
    
    OutParamsBase(adapted_param_df)
    OutParamAssChrom(adapted_param_df)


def test_get_asschrom_results(get_acr):
    '''
    Simply test whether the member objects of AssChromResults are initialized.
    '''
    assert any(get_acr.param_df)
    assert any(get_acr.ptable)
    assert any(get_acr.unmixed)
    
    
    print(f"\n{get_acr.param_df}")
    print(f"\n{get_acr.ptable}")
    print(f"\n{get_acr.unmixed}")
    
@pytest.fixture
def datapath():
    return "tests/test_data/test_assessment_chrom.csv"


@pytest.fixture
def testsignal(datapath: Literal['tests/test_data/test_many_peaks.csv']) -> pt.DataFrame[SignalDFIn]:
    data = pd.read_csv(datapath)
    assert isinstance(data, pd.DataFrame)

    data = data.rename({"x": "time", "y": "amp"}, axis=1)

    return typing.cast(pt.DataFrame[SignalDFIn], data)


@pytest.fixture
def chm():
    return Chromatogram(viz=False)


@pytest.fixture
def time(testsignal: pt.DataFrame[SignalDFIn]):
    assert isinstance(testsignal, pd.DataFrame)
    assert isArrayLike(testsignal.time)
    return testsignal.time.values


@pytest.fixture
def timestep(chm: Chromatogram, time: npt.NDArray[np.float64]) -> float:
    timestep = chm.compute_timestep(time)
    assert timestep
    assert isinstance(timestep, float)
    assert timestep > 0

    return timestep

def test_timestep(timestep):
    print(timestep)

@pytest.fixture
def amp(testsignal: DataFrame[SignalDFIn]):
    assert isArrayLike(testsignal.amp)
    return testsignal.amp.values


@pytest.fixture
def windowsize():
    return 5


@pytest.fixture
def bcorr_col(intcol: str) -> str:
    return intcol + "_corrected"


@pytest.fixture
def timecol():
    return "time"


@pytest.fixture
def intcol():
    return "signal"

@pytest.fixture
def bcorr(
    chm: Chromatogram,
    amp: npt.NDArray[np.float64],
    timestep: np.float64,
    windowsize: int,
):
    bcorr = chm._baseline.correct_baseline(amp,
                                            timestep,
                                            windowsize,
                                            verbose=False,
                                            )[0]

    return bcorr
    
@pytest.fixture
def amp_cn(
    chm: Chromatogram, bcorr: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    ``int_cn` has the base data as the first element of the namespace then the process
    in order. i.e. intensity: [corrected, normalized]
    """
    int_cn = chm._findwindows.normalize_intensity(bcorr)

    assert any(int_cn)
    assert isArrayLike(int_cn)
    assert np.min(int_cn >= 0)
    assert np.max(int_cn <= 1)

    return int_cn


@pytest.fixture
def peak_df(
    chm: Chromatogram,
    time: npt.NDArray[np.float64],
    amp_cn: npt.NDArray[np.float64],
) -> pt.DataFrame[OutPeakDF_Base]:
    peak_df = chm._findwindows.peak_df_factory(time, amp_cn)

    return peak_df


@pytest.fixture
def norm_amp(
    chm: Chromatogram,
    bcorr: npt.NDArray[np.float64],
):
    norm_int = chm._findwindows.normalize_intensity(bcorr)
    return norm_int


@pytest.fixture
def signal_df(
    chm: Chromatogram,
    amp: npt.NDArray[np.float64],
) -> pt.DataFrame:
    signal_df = chm._findwindows.signal_df_factory(amp)
    return signal_df


@pytest.fixture
def window_df(
    chm: Chromatogram,
    signal_df: pt.DataFrame,
    peak_df: pt.DataFrame[OutPeakDF_Base],
):
    window_df = chm._findwindows.window_df_factory(
        signal_df,
        peak_df,
    )

    return window_df


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
        testsignal: pt.DataFrame[SignalDFIn],
        valid_time_windows: list[list[int]],
    ):
        """
        test `crop()` by applying a series of valid time windows then testing whether all values within the time column fall within that defined range.
        """

        for window in valid_time_windows:
            assert len(window) == 2

            # copy to avoid any leakage across instances

            testsignal = chm.crop(testsignal, time_window=window)

            leq_mask = testsignal.time >= window[0]
            geq_mask = testsignal.time <= window[1]

            assert (leq_mask).all(), f"{testsignal[leq_mask].index}"
            assert (geq_mask).all(), f"{testsignal[geq_mask].index}"

        return None

    def test_testsignal(self, testsignal: pt.DataFrame[SignalDFIn]) -> None:
        SignalDFIn(testsignal)

        return None

    def test_crop_invalid_time_window(
        self,
        chm: Chromatogram,
        testsignal: pt.DataFrame[SignalDFIn],
        invalid_time_window: list[list[int]],
    ):
        for window in invalid_time_window:
            try:
                chm.crop(testsignal, window)

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
    def amp(self, testsignal: DataFrame[SignalDFIn])-> npt.NDArray[np.float64]:
        return testsignal.amp.to_numpy(dtype=np.float64)

    @pytest.fixture
    def compressed_amp(
        self, chm: Chromatogram, amp: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # intensity raw compressed
        intensity_rc = chm._baseline.compute_compressed_signal(amp)

        assert intensity_rc

        return intensity_rc

    def test_get_tform(self, chm: Chromatogram, amp: npt.NDArray[np.float64]):
        tform = chm._baseline.compute_compressed_signal(amp)

        assert np.all(tform)
        assert isinstance(tform, np.ndarray)

    def test_compute_inv_tfrom(
        self,
        chm: Chromatogram,
        amp: npt.NDArray[np.float64],
    ) -> None:
        chm._baseline.compute_inv_tform(amp)

    def test_correct_baseline(
       self,
        amp: npt.NDArray[np.float64],
        chm: Chromatogram,
        time: npt.NDArray[np.float64],
        timestep,
    ) -> None:
        # pass the test if the area under the corrected signal is less than the area under the raw signal
        x_start = time[0]
        x_end = time[-1]
        n_x = len(time)
        
        # add a preset baseline to ensure that correction behavior is within expected scale
        
        x = np.linspace(x_start,x_end,n_x)
        
        from scipy import stats
        
        skew = 1
        loc = x_end*0.3
        scale = x_end*0.3
        skewnorm = stats.skewnorm(skew, loc=loc, scale=scale)
        
        y = skewnorm.pdf(x) *np.max(amp)**2
        
        added_baseline = amp+y
        
        bcorr = chm._baseline.correct_baseline(added_baseline,
                                              timestep,
                                              
                                              )[0]
        
        baseline_auc = integrate.trapezoid(added_baseline, time)
        
        bcorr_auc = integrate.trapezoid(bcorr, time)

        assert baseline_auc > bcorr_auc


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
    def all_ranges_mask(self, chm: Chromatogram, all_ranges: list[npt.NDArray[np.int64]]):
        mask = chm._findwindows.mask_subset_ranges(all_ranges)

        assert len(mask) > 0
        assert any(mask == True)

        return mask

    @pytest.fixture
    def ranges_with_subset(self,
                           chm: Chromatogram,
                           all_ranges_mask: npt.NDArray[np.bool_],
                           all_ranges:
                               list[npt.NDArray[np.int64]]):
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

    test_signal_df_kwargs = {
        "argnames": ["datapath", "schema"],
        "argvalues": [
            (manypeakspath, OutSignalDF_ManyPeaks),
            (
                asschrompath,
                OutSignalDF_Base,
            ),
        ],
    }

    @pytest.mark.parametrize(**test_signal_df_kwargs)
    def test_signal_df(
        self,
        signal_df: DataFrame[OutSignalDF_Base],
        schema,
    ) -> None:
        schema(signal_df)

    test_peak_df_kwargs = {
        "argnames": ["datapath", "schema"],
        "argvalues": [
            (
                manypeakspath,
                OutPeakDF_ManyPeaks
            ),
            (
                asschrompath,
                OutPeakDF_AssChrom,
            ),
        ],
    }
    @pytest.mark.parametrize(**test_peak_df_kwargs)
    def test_peak_df(
        self,
        peak_df: pt.DataFrame,
        schema,
    ):
        try:
            schema(peak_df)
        except Exception as e:
            raise RuntimeError(
                f"{e}\n"
                f"\n{schema}\n"
                f"\n{peak_df.to_markdown()}\n"
            )
    
    def test_peak_df_viz(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        timestep
    ):

        '''
        overlay the plot metrics on the peaks
        '''
        plt.style.use('ggplot')
        
        def pplot(peak_df, signal_df):
        # signal
        
            plt.plot(signal_df['time_idx'], signal_df['norm_amp'], c='blue')
            
            # peaks
            
            plt.scatter(
                peak_df['time_idx'], peak_df['norm_amp'], c='red')
            
            # left and right bases of each width measurement
            # the left and right values are values of the x axis, so simply plot them with the 'height' as the y value
            
            # whh measurement
            
            plt.hlines(peak_df['whhh'],
                       peak_df['whh_left'],
                       peak_df['whh_right'],
                       label='whh')
            
            # 'relative height' width
            
            plt.hlines(
                # peak_df['rl_wh'],
                [-0.1]*len(peak_df),
                peak_df['rl_left'],
                peak_df['rl_right'],
                label='rl width',
                color='green'
                
            )
            plt.legend()
            plt.show()
            
            return None
        
        peak_df = (
            peak_df
            .set_index('time_idx')
            .join(
                signal_df
                .set_index('time_idx'),
                how='left',
            )
           .reset_index()
           .set_index('peak_idx')
           .reset_index()
        ) # type: ignore
        
        pplot(peak_df, signal_df)
        
        print(signal_df.head())
        
        print(peak_df)
        
        print(timestep)
        
    @pytest.fixture
    def window_df(
        self,
        chm: Chromatogram,
        signal_df: pt.DataFrame,
        peak_df: pt.DataFrame,
    ) -> pt.DataFrame:
        window_df = chm._findwindows.window_df_factory(
            signal_df,
            peak_df,
        )
        return window_df

    test_window_df_kwargs = {
        "argnames": ["datapath", "schema"],
        "argvalues": [
            (manypeakspath, OutWindowDF_ManyPeaks),
            (
                asschrompath,
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
            raise RuntimeError(
                f"{schema}\n"
                f"{e}\n"
                f"{window_df.describe()}\n"
            )

        return None

    test_windows_plot_kwargs = {
        "argnames": ["datapath"],
        "argvalues": [(manypeakspath,), (asschrompath,)],
    }

    @pytest.mark.parametrize(**test_windows_plot_kwargs)
    def test_windows_plot(
        self,
        chm: Chromatogram,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
    ) -> None:
    
        # signal overlay

        fig, ax = plt.subplots(1)
        
        peak_signal_join=(
            peak_df
            .set_index("time_idx")
            .join(
                signal_df
                .set_index("time_idx"),
                how='left',
                validate='1:1'
            )
            .reset_index()
        )
        
        # the signal
        
        ax.plot(signal_df.time_idx, signal_df.norm_amp, label='signal')
        
        pwtable = chm._findwindows.window_df_pivot(window_df)

        def signal_window_overlay(
            ax,
            signal_df: pt.DataFrame[OutSignalDF_Base],
            pwtable: pd.DataFrame,
        ) -> None:
            """
            Create an overlay of the signal and the windows.
            """

            set2 = mpl.colormaps["Set2"].resampled(pwtable.groupby("window_idx").ngroups)
            
            for id, window in pwtable.groupby("window_idx"):
                
                anchor_x = window["min"].values[0]
                anchor_y = 0
                width = window["max"].values[0] - window["min"].values[0]
                max_height = signal_df.norm_amp.max()
                
                rt = Rectangle(
                    xy=(anchor_x, anchor_y),
                    width=width,
                    height=max_height,
                    color=set2.colors[int(id) - 1], #type: ignore
                )

                ax.add_patch(rt)            

            return ax

        signal_window_overlay(ax, signal_df, pwtable)

        # the peaks
        
        ax.scatter(peak_signal_join.time_idx, peak_signal_join.norm_amp, label='peaks', color='red')
        
        # now draw the interpolations determining the peak width
        
        
        fig.show()
        plt.show()
    
    
        
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
        bcorr: npt.NDArray[np.float64],
    ) -> None:
        if bcorr.ndim != 1:
            raise ValueError

        assert len(time) > 0
        assert len(bcorr) > 0

        try:
            chm._findwindows.profile_peaks_assign_windows(
                time,
                bcorr,
                timestep,
            )
        except Exception as e:
            raise RuntimeError(e)

@pytest.fixture
def default_bounds(
    chm: Chromatogram,
    p0: pt.DataFrame[OutInitialGuessBase],
    window_df: pt.DataFrame[OutWindowDF_Base],
    peak_df: pt.DataFrame[OutPeakDF_Base],
    timestep: np.float64,
                    ):
    default_bounds = chm._deconvolve.dataprepper.default_bounds_factory(
            p0,
            window_df,
            peak_df,
            timestep,
            )
    return default_bounds
    
@pytest.fixture
def p0(
    chm: Chromatogram,
    signal_df: pt.DataFrame[OutSignalDF_Base],
    peak_df: pt.DataFrame[OutPeakDF_Base],
    window_df: pt.DataFrame[OutWindowDF_Base],
)->pt.DataFrame[OutInitialGuessBase]:
    
    p0_df = chm._deconvolve.dataprepper.p0_factory(
                signal_df,
                peak_df,
                window_df,
        )
    return p0_df

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
        chm: Chromatogram,
        signal_df: pt.DataFrame[OutSignalDF_Base]
    ) -> None:
        """
        find the integration range in time units for the given user input: note: note sure
        how to ttest this currently..

        TODO: add better test
        """

        tr = chm._deconvolve.dataprepper.find_integration_range(
            signal_df['time_idx'], #type: ignore
            [30, 40],
            )
        
        assert pd.Series(tr).isin(signal_df['time_idx']).all()


    @pytest.mark.parametrize(
        [
            'datapath',
            'schema'
        ],
        [
            (
                manypeakspath,
                OutInitialGuessManyPeaks,
            ),
            (
                asschrompath,
                OutInitialGuessAssChrom,
            )
        ]
    )
    def test_p0_factory(
        self,
        p0: pt.DataFrame[OutInitialGuessBase],
        schema,
    ):
        '''
        Test the initial guess factory output against the dataset-specific schema.
        '''
        
        schema(p0)
        
        return None
    

        
    @pytest.mark.parametrize(
        [
            'datapath',
            'schema'
        ],
        [
            # (
            #     manypeakspath,
            #     OutDefaultBoundsManyPeaks,
            # ),
            (
                asschrompath,
                OutDefaultBoundsAssChrom,
            )
        ]
    )
    def test_default_bounds_factory(
        self,
        default_bounds: pt.DataFrame[OutDefaultBoundsBase],
        schema,
    )->None:
        '''
        Define default bounds schemas
        '''
        
        schema(default_bounds)
        
        
        return None
    
'''
2023-12-08 10:08:47

Since the trivial inputs work, we need to unit test p optimizer to expose the failcase data.
'''
@pytest.fixture
def param_df(
    chm: Chromatogram,
    p0: pt.DataFrame[OutInitialGuessBase],
    default_bounds: pt.DataFrame[OutDefaultBoundsBase],
)->pt.DataFrame[OutParamsBase]:

    param_df = chm._deconvolve.dataprepper._param_df_factory(
        p0,
        default_bounds
    )
    
    return param_df

@pytest.fixture
def windowed_signal_df(
    chm: Chromatogram,
    signal_df: pt.DataFrame[OutSignalDF_Base],
    window_df: pt.DataFrame[OutWindowDF_Base],
                                )->pt.DataFrame[OutWindowedSignalBase]:
    
    '''
    test the output against a defined schema.
    '''
    
    
    windowed_signal_df = chm._deconvolve.dataprepper._window_signal_df(
        signal_df,
        window_df
    )
    
    return windowed_signal_df.pipe(pt.DataFrame[OutWindowedSignalBase]) # type: ignore

@pytest.fixture
def popt_df(
    chm: Chromatogram,
    windowed_signal_df:pt.DataFrame[OutWindowedSignalBase],
    param_df: pt.DataFrame[OutParamsBase],
):
    popt_df = chm._deconvolve._popt_factory(
        windowed_signal_df,
        param_df
    )
    return popt_df

@pytest.mark.parametrize(
    ['params','x'],
    [
        ([10, 5, 2, 0, 20, 10, 4, 2], np.arange(0,30,1, dtype=np.float64))
     ],
)
class TestingCurveFit:
    @pytest.fixture
    def y(
        self,
        chm: Chromatogram,
        params,
        x
    ):
        '''
        Need:
        
        - [ ] time axis
        - [ ] params.
        '''
        
        results = chm._deconvolve._fit_skewnorms(x, *params)
    
        return results
    
    def test_fit_skewnorms(
        self,
        y)->None:
        '''
        simply test if y is able to execute successfully
        '''
        
        try:
            assert all(y)
        except Exception as e:
            raise RuntimeError(e)
    
    def test_curve_fit(
        self,
        chm: Chromatogram,
        params,
        x,
        y):
        '''
        test if optimize.curve_fit operates as expected
        '''
        from scipy import optimize
        
        func = chm._deconvolve._fit_skewnorms
        
        try:
            popt, _ = optimize.curve_fit(
                func,
                x,
                y,
                params
            )
        except Exception as e:
            raise RuntimeError(e)
        
        popt = popt.reshape(2,4)
        
        window_dict = {}
        for peak_idx, p in enumerate(popt):
            window_dict[f"peak_{peak_idx + 1}"] = {
                "amplitude": p[0],
                "retention_time": p[1],
                "scale": p[2],
                "alpha": p[3],
                "area": chm._deconvolve._compute_skewnorm(x, *p).sum(),
                "reconstructed_signal": chm._deconvolve._compute_skewnorm(
                    x, *p
                ),
            }
        
        
@pytest.fixture
def xdata(
    signal_df,
):
    return signal_df['time_idx']
    
@pytest.fixture
def unmixed_df(
    chm: Chromatogram,
    xdata,
    stored_popt,
):
    unmixed_df = chm._deconvolve._reconstruct_peak_signal(
        xdata,
        stored_popt
    )
    return unmixed_df
    
@pytest.fixture
def peak_report(
    chm: Chromatogram,
    stored_popt: OutPoptBase,
    unmixed_df: OutReconDFBase,
    timestep: np.float64,
    
)-> OutPeakReportBase:
    peak_report = chm._deconvolve.compile_peak_report(
        stored_popt,
        unmixed_df,
        timestep,
    )
    return peak_report.pipe(pt.DataFrame[OutPeakReportBase]) # type: ignore

class TestDeconvolver(TestSuperClass):
    manypeakspath = "tests/test_data/test_many_peaks.csv"
    asschrompath = "tests/test_data/test_assessment_chrom.csv"
    
    
    @pytest.mark.parametrize(
        ['datapath','schema'],
        [
        (manypeakspath,OutWindowedSignalManyPeaks,),
        (asschrompath,OutWindowedSignalAssChrom,),
         ]
    )
    def test_windowed_signal_df(
        self,
        windowed_signal_df,
        schema,
    )->None:
        schema(windowed_signal_df)

    
    @pytest.mark.parametrize(
        ['datapath','schema'],
        [
        # (manypeakspath,OutParamManyPeaks,),
        (asschrompath,OutParamAssChrom,),
         ]
    )
    def test_param_df_factory(
        self,
        param_df: pt.DataFrame[OutParamsBase],
        schema,
                              )->None:
        
        print(f"\n{param_df}")        
        try:
            schema(param_df)
        except Exception as e:
            "if its a schema error, update the schema with these values if appropriate"
            for col in param_df:
                print(col)
                print(param_df[col].value_counts().index.tolist())
            raise ValueError(e)
        
        
        return None
    
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
            param_df,
        )
        return params
    
    @pytest.mark.parametrize(
        ['datapath','window'],
        [
            (
                manypeakspath, 1,
            ),
            (
                asschrompath, 1,
            ),
            (
                asschrompath, 2,
            )
        ]
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
        
    @pytest.fixture
    def popt_parqpath(self):
        '''
        Intended to be used to store a popt df as it is computationally expensive to deconvolute many-peaked windows
        '''
        return "/Users/jonathan/hplc-py/tests/jonathan_tests/popt.parquet"
    
    def test_popt_to_parquet(
        self,
        stored_popt,
        popt_parqpath
    ):
        '''
        A function used to produce a parquet file of a popt df. I suppose it itself acts as a test, and means that whenever i run the full suite the file will be refreshed.
        '''
        
        stored_popt.to_parquet(popt_parqpath)
    
    @pytest.fixture()
    def stored_popt(
        self,
        popt_parqpath
    ):
        '''
        Read the stored popt_df, short circuiting the slow optimization process
        '''
        return pd.read_parquet(popt_parqpath)
    
    @pytest.mark.parametrize(
        ['datapath',
         'schema'
         ],
        [
            (
                asschrompath,
                OutPoptAssChrom,
                
            )
        ]
    )
    def test_popt_factory(
        self,
        popt_df,
        schema,
    ):
        '''
        TODO:
        - [ ] define dataset specific schemas
        - [ ] identify why algo needs more than 1200 iterations to minimize mine vs 33 for main
        - [ ] testing with the main adapted param_df, 24 iterations for the first window, 21 for the second. Whats the difference? 
        '''
        
        schema(popt_df)
        
        return None
        
    '''
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
    '''
    
    def test_popt_factory_main_params(
        self,
        chm: Chromatogram,
        get_acr,
        
        windowed_signal_df,
        
    ):
        
        print(get_acr.adapted_param_df)
        
        # popt_df = chm._deconvolve._popt_factory(windowed_signal_df,
        #                               get_acr.adapted_param_df,
        #                               )
        print(popt_df)  
        
    
    def test_reconstruct_peak_signal(
        self,
        unmixed_df: OutReconDFBase,
    )-> None:
        '''
        TODO:
        - [ ] establish schemas for the datasets and rewrite the test for them
        '''
        
        OutReconDFBase(unmixed_df)
    
    @pytest.mark.parametrize(
            [
                # 'datapath',
            'schema'
            ],
            [
                (
                    # asschrompath,
                    OutPeakReportAssChrom,
                )
            ]
        )
    def test_peak_report(
        self,
        peak_report: OutPeakReportBase,
        schema,
    ):  
        schema(peak_report)

    
    @pytest.mark.parametrize(
            ['datapath',
            # 'schema'
            ],
            [
                (
                    asschrompath,
                    # OutPoptAssChrom,
                    
                )
            ]
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
            signal_df,
            peak_df,
            window_df,
            timestep
        )
        
class TestFitPeaks:
    '''
    test the `fit_peaks` call, which performs the overall process to unmix the peaks and
    provide a peak table
    '''

    @pytest.fixture
    def fit_peaks(self,
                       chm: Chromatogram,
                       testsignal: pt.DataFrame[SignalDFIn],
                       ):
        
        chm.load_data(
            testsignal)
        
        popt_df, unmixed_df = chm.fit_peaks()
        
        return popt_df, unmixed_df
    
    def test_fit_peaks(
        self,
        popt_df,
        unmixed_df,
    ):
    
        assert any(popt_df)
        assert any(unmixed_df)
        
        return None
    
    @pytest.fixture
    def popt_df(self,
                fit_peaks):
        return fit_peaks[0]
    
    @pytest.fixture
    def unmixed_df(self,
                 fit_peaks):
        return fit_peaks[1]
    
    def test_compare_timesteps(
        self,
        get_acr,
        timestep,
    ):
    
        
        assert get_acr.timestep == timestep, f"timesteps are not equal. {get_acr.timestep, timestep}"
    
    def test_compare_param_dfs(
        self,
        param_df,
        get_acr
    ):
        '''
        method for comparing the adapted param df and my param df    
        '''
        
        # print(
        #         param_df.copy().compare(get_acr.adapted_param_df)
        # )
        print(f"\n{param_df}")
        print(f"\n{get_acr.adapted_param_df}")
        
        print(f"\n{param_df.dtypes}")
        print(f"\n{get_acr.adapted_param_df.dtypes}")
    
    def test_compare_opt_params(
        self,
        param_df,
        timestep,
        get_acr,
    ):
        # print(f"\n{popt_df}")
        
        print_main_df = (get_acr
                         .param_df
                        #  .query("param=='scale'")
                         )
        print(f"\n{print_main_df}")
        
        num_cols = ['p0','lb','ub']
        mask = (param_df['param']=='loc')|(param_df['param']=='whh')
        
        param_df.loc[mask,num_cols]=param_df.loc[mask,num_cols]*timestep
        
        print_my_df = (param_df
                    #    .query("param=='whh'")
                    #    .astype(float, errors='ignore')
                       )
        print(f"\n{print_my_df}")
    '''
    
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
    '''    
        
class TestShow:
    '''
    Test the Show class methods
    '''
    
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
        return chm._deconvolve.deconvolve_peaks(
            signal_df,
            peak_df,
            window_df,
            timestep
        )
        
    @pytest.fixture
    def popt_df(
        self,
        decon_out
    ):
        return decon_out[0]

    @pytest.fixture
    def popt_df(
        self,
        decon_out
    ):
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
        chm.show.plot_reconstructed_signal(
            unmixed_df,
            fig_ax[1]
        )
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
        
    