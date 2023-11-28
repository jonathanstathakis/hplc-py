import pytest

import pandas as pd
import pandera.typing as pt
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import copy
from hplc_py.quant import Chromatogram
from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike

from hplc_py.hplc_py_typing.hplc_py_typing import WidthDF, WindowedSignalDF

plt.style.use('bmh')

@pytest.fixture
def int_cn(chm: Chromatogram, bcorr:npt.NDArray[np.float64])->npt.NDArray[np.float64]:
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

@ pytest.fixture
def prominence():
    return 0.01

@pytest.fixture
def peak_idx(chm: Chromatogram,
              int_cn:pt.Series[float],
              prominence: int|float,
              )->pt.Series:
    
    peak_idx = chm.findwindows.compute_peak_idx(
        int_cn,
        prominence
        )
    
    assert any(peak_idx)
    assert len(peak_idx)>0
    assert all(peak_idx>0)
    
    return peak_idx

@pytest.fixture
def width_df(chm: Chromatogram,
                              bcorr: pt.Series[float],
                              peak_idx: pt.Series[int],
                              ):
    
    width_df = chm.findwindows.build_width_df(
                                         peak_idx,
                                         bcorr,
                                         )
    return width_df

@pytest.fixture
def norm_int(chm: Chromatogram,
             bcorr: npt.NDArray[np.float64],
             ):
    norm_int = chm.findwindows.normalize_intensity(bcorr)
    return norm_int
    
@pytest.fixture
def all_ranges(
    chm:Chromatogram,
    norm_int:pt.Series[float],
    width_df:pt.DataFrame[WidthDF],
):
    ranges = chm.findwindows.compute_individual_peak_ranges(norm_int, width_df['left'],width_df['right'])
    
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

@pytest.fixture
def windowed_df(chm: Chromatogram,
              norm_int: npt.NDArray[np.float64],
              time: npt.NDArray[np.float64],
              width_df: pt.DataFrame[WidthDF],
              ):
    window_df = chm.findwindows.window_signal_df(
        norm_int,
        time,
        width_df.left.to_numpy(np.float64),
        width_df.right.to_numpy(np.float64),
    )
    
    return window_df

def test_peak_idx(peak_idx):
    assert any(peak_idx)

def test_norm_int(norm_int:npt.NDArray[np.float64])->None:
    assert len(norm_int)>0
    assert np.min(norm_int)==0
    assert np.max(norm_int)==1

def test_width_df(width_df):
    assert isinstance(width_df, pd.DataFrame)
    assert not width_df.empty

def test_window_df(windowed_df):
    
    assert not windowed_df.empty
    
    print(
        windowed_df
    )

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
    
    return validated_ranges

def test_construct_dict_of_window_dicts(chm: Chromatogram, window_df: pt.DataFrame[WindowedSignalDF], width_df: pt.DataFrame[WidthDF])->None:
    
    window_dicts = chm.findwindows.construct_dict_of_window_dicts(window_df,width_df,)

    assert any(window_dicts)
    
    print(window_dicts)
    
    return None

def test_find_windows(chm: Chromatogram,
                      time: pt.Series,
                      timestep: float,
                      bcorr: pt.Series,)->None:
    
    if bcorr.ndim!=1:
        raise ValueError
    
    chm.findwindows.assign_windows(
                        time,
                        bcorr,
                        timestep,
                        
                        )