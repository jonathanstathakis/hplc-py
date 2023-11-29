"""
TODO:

- Finish defining the test schemas and use them where appropriate
- further define the tests.
"""

import pytest

import pandas as pd
import pandera as pa
import pandera.typing as pt
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import copy
from hplc_py.quant import Chromatogram
from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike

from hplc_py.hplc_py_typing.hplc_py_typing import PeakDF, TestOutPeakDF, BaseWindowedSignalDF, TestOutWindowedSignalDF, TestAugmentedFrameWidthMetrics

plt.style.use('bmh')

@pytest.fixture
def amp_cn(chm: Chromatogram, bcorr:npt.NDArray[np.float64])->npt.NDArray[np.float64]:
    '''
    `int_cn` has the base data as the first element of the namespace then the process 
    in order. i.e. intensity: [corrected, normalized]
    '''
    int_cn = chm.findwindows.normalize_intensity(bcorr)
    
    assert any(int_cn)
    assert isArrayLike(int_cn)
    assert np.min(int_cn>=0)
    assert np.max(int_cn<=1)
    
    return int_cn

@pytest.fixture
def peak_df(chm: Chromatogram,
              amp_cn:npt.NDArray[np.float64],
              )->npt.NDArray[np.float64]:
    
    peak_df = chm.findwindows.build_peak_df(amp_cn)
    
    return peak_df.pipe(pt.DataFrame[PeakDF])

@pytest.fixture
def norm_amp(chm: Chromatogram,
             bcorr: npt.NDArray[np.float64],
             ):
    norm_int = chm.findwindows.normalize_intensity(bcorr)
    return norm_int
    
@pytest.fixture
def all_ranges(
    chm:Chromatogram,
    norm_amp:npt.NDArray[np.float64],
    peak_df:pt.DataFrame[PeakDF],
):
    ranges = chm.findwindows.compute_individual_peak_ranges(
        norm_amp,
        peak_df['rl_left'],
        peak_df['rl_right']
        )
    
    for range in ranges:
        assert all(range)
        
    assert len(ranges)>0
    return ranges

def test_all_ranges(all_ranges):
    for range in all_ranges:
        assert all(range)

@pytest.fixture
def all_ranges_mask(chm, all_ranges):
    mask = chm.findwindows.mask_subset_ranges(all_ranges)
    
    assert len(mask)>0
    assert any(mask==True)
    
    return mask

@pytest.fixture
def ranges_with_subset(chm, all_ranges_mask, all_ranges):
    
    new_ranges = copy.deepcopy(all_ranges)
    
    # for the test data, there are no subset ranges, thus prior to setup all
    # values in the mask will be true
    
    new_ranges.append(new_ranges[-1])
    
    return new_ranges

@pytest.fixture
def ranges_with_subset_mask(chm, ranges_with_subset):
    
    new_mask = chm.findwindows.mask_subset_ranges(ranges_with_subset)
    assert any(new_mask==False)
    return new_mask

def test_validate_ranges(chm,
                         all_ranges,
                         all_ranges_mask,
                         ranges_with_subset,
                         ranges_with_subset_mask,
                         ):
    """
    Bring together all the setup for the range validation for testing
    """
    for i in [all_ranges, ranges_with_subset]:
        for j in i:
            assert any(j)
    
    assert any(all_ranges_mask)
    assert any(ranges_with_subset_mask)
    
    assert ranges_with_subset_mask[-1]==False
    
    assert all(all_ranges_mask)
    
    all_ranges_validated = chm.findwindows.validate_ranges(all_ranges, all_ranges_mask)
    subset_ranges_validated = chm.findwindows.validate_ranges(ranges_with_subset, ranges_with_subset_mask)
    
    assert len(all_ranges)>0
    assert len(all_ranges_validated)>0
    
    for range in all_ranges_validated:
        assert any(range)
            
    
    assert len(all_ranges_validated)==len(all_ranges)
    assert len(subset_ranges_validated)<len(ranges_with_subset)
    
def test_mock_validate_ranges(ranges_with_subset, ranges_with_subset_mask):

    validated_ranges = [0]
    
    for i, r in enumerate(ranges_with_subset):
        if ranges_with_subset_mask[i]:
            validated_ranges.append(r)
    
    return None

@pytest.fixture
def windowed_signal_df(
              chm: Chromatogram,
              amp: npt.NDArray[np.float64],
              norm_amp: npt.NDArray[np.float64],
              time: npt.NDArray[np.float64],
              peak_df: pt.DataFrame[PeakDF],
              ):
    window_df = chm.findwindows.window_signal_df(
        amp,
        norm_amp,
        time,
        peak_df.rl_left.to_numpy(np.float64),
        peak_df.rl_right.to_numpy(np.float64),
    )
    
    return window_df

def test_norm_amp(norm_amp:npt.NDArray[np.float64])->None:
    assert len(norm_amp)>0
    assert np.min(norm_amp)==0
    assert np.max(norm_amp)==1

def test_peak_df(peak_df: pt.DataFrame[PeakDF]):
    TestOutPeakDF(peak_df)
    
def test_maxima_idx(norm_amp: npt.NDArray[np.float64]):
    
    from scipy.signal import find_peaks
    
    maxima, _ = find_peaks(norm_amp)
    
def test_window_df(windowed_signal_df):
    
    TestOutWindowedSignalDF(windowed_signal_df)

def test_create_augmented_df(chm: Chromatogram,
                                        windowed_signal_df: pt.DataFrame[BaseWindowedSignalDF],
                                        peak_df: pt.DataFrame[PeakDF],
                                        timestep: np.float64,
                                        )->None:
    
    
    # seperate the below into individual test schemas
    
    augmented_df = (chm.findwindows
                    .create_augmented_df(
        peak_df,
        windowed_signal_df,
        timestep            
        )       
    )
        
    try:
        augmented_df.pipe(TestAugmentedFrameWidthMetrics)
    except Exception as e:
        raise ValueError(f"schema error: {e}")
    
    return None

def test_assign_windows(chm: Chromatogram,
                      time: npt.NDArray[np.float64],
                      timestep: float,
                      bcorr: npt.NDArray[np.float64],)->None:
    
    if bcorr.ndim!=1:
        raise ValueError
    
    assert len(time)>0
    assert len(bcorr)>0
    
    chm.findwindows.profile_peaks_assign_windows(
                        time,
                        bcorr,
                        timestep,
                        )