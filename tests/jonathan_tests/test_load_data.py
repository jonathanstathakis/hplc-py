import pytest

import pandas as pd
import pandera as pa
import pandera.typing as pt
import numpy as np
import matplotlib.pyplot as plt
import copy

from hplc_py.quant import Chromatogram
from hplc_py.hplc_py_typing.hplc_py_typing import SignalDFIn

def test_load_chm(chm, testsignal)->None:
    if not isinstance(chm, Chromatogram):
        raise TypeError("chm is not a Chromatogram object")
        
    chm.load_data(testsignal)
    
    if not all(chm._internal_df):
        raise ValueError('df not initiated')
    else:
        if chm._internal_df.empty:
            raise ValueError('df is empty')
        
@pytest.fixture
def timestep(chm, testsignal)-> float:
    timestep = chm.compute_timestep(testsignal.time)
    assert timestep, "timestep not initialized"
    
    assert timestep > 0, "timestep unrealistic"
    return timestep

def test_timestep(timestep):
    assert timestep
    
@pytest.fixture
def valid_time_windows():
    return [[0,5],[5,15]]

@pytest.fixture
def invalid_time_window():
    return [[15,5]]

def test_crop_valid_time_windows(chm: Chromatogram,
                                 testsignal: pt.DataFrame[SignalDFIn],
                                 valid_time_windows: list[list[int]],
                                 ):
    '''
    test `crop()` by applying a series of valid time windows then testing whether all values within the time column fall within that defined range.
    '''
    
    for window in valid_time_windows:
            
            assert len(window)==2
            
            # copy to avoid any leakage across instances
            
            df = testsignal.copy(deep=True).pipe(pt.DataFrame[SignalDFIn])
            
            df = chm.crop(df,
                          time_window=window)
            
            leq_mask = df.time>=window[0]
            geq_mask = df.time<=window[1]
            
            assert (leq_mask).all(), f"{df[leq_mask].index}"
            assert (geq_mask).all(), f"{df[geq_mask].index}"
    
    return None

def test_crop_invalid_time_window(chm: Chromatogram,
                                testsignal:pt.DataFrame[SignalDFIn],
                                invalid_time_window:list[list[int]]
                                    ):
    for window in invalid_time_window:
        try:
            
            chm.crop(testsignal, window)
            
        except RuntimeError as e:
            continue