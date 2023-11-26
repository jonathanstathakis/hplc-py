import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from hplc_py.quant import Chromatogram
from hplc_py.hplc_py_typing.hplc_py_typing import checkArrayLike
        
@pytest.fixture
def timestep(chm, time_array)-> float:
    timestep = chm.get_timestep(time_array)
    assert timestep, "timestep not initialized"
    
    assert timestep > 0, "timestep unrealistic"
    return timestep

def test_timestep(timestep, time_array, loaded_chm):
    assert timestep == loaded_chm.get_timestep(time_array), "calculated timesteps differ"
    
@pytest.fixture
def valid_time_windows():
    return [[0,5],[5,15]]

@pytest.fixture
def invalid_time_window():
    return [[15,5]]

def test_crop_valid_time_windows(loaded_chm: Chromatogram,
                                 valid_time_windows,
                                 timecol,
                                 ):
    '''
    test `crop()` by applying a series of valid time windows then testing whether all values within the time column fall within that defined range.
    '''
    
    for window in valid_time_windows:
            
            assert len(window)==2
            
            # copy to avoid any leakage across instances
            lchm = copy.deepcopy(loaded_chm)
            df = lchm.df.copy(deep=True)
            
            df = lchm.crop(df, timecol, window)
            
            leq_mask = df[timecol]>=window[0]
            geq_mask = df[timecol]<=window[1]
            assert (leq_mask).all(), f"{df[leq_mask].index}"
            assert (geq_mask).all(), f"{df[geq_mask].index}"
            
            del lchm

def test_crop_invalid_time_window(loaded_chm: Chromatogram, invalid_time_window, timecol):
    for window in invalid_time_window:
        try:
            loaded_chm.crop(loaded_chm.df, timecol, window)
        except RuntimeError as e:
            continue