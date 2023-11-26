import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from hplc_py.quant import Chromatogram
from hplc_py.hplc_py_typing.hplc_py_typing import checkArrayLike

@pytest.fixture
def testdata_path():
    return "tests/test_data/test_many_peaks.csv"

@pytest.fixture
def testdata(testdata_path)-> pd.DataFrame:
    data = pd.read_csv(testdata_path)
    
    assert isinstance(data, pd.DataFrame)
    
    return data
    
@pytest.fixture
def chm():
    return Chromatogram()

@pytest.fixture
def time_series(testdata):
    assert isinstance(testdata, pd.DataFrame)
    assert checkArrayLike(testdata.x)
    return testdata.x.values

@pytest.fixture
def signal_series(testdata):
    assert checkArrayLike(testdata.y)
    return testdata.y.values

@pytest.fixture
def timecol():
    return 'time'

@pytest.fixture
def signal_col():
    return 'signal'

@pytest.fixture
def loaded_chm(chm, time_series, signal_series)->Chromatogram:
    if not isinstance(chm, Chromatogram):
        raise TypeError("chm is not a Chromatogram object")
    
    if not checkArrayLike(time_series):
        raise TypeError("x must be ArrayLike")
    
    if not checkArrayLike(signal_series):
        assert TypeError(f"y must be ArrayLike, but passed {type(signal_series)}")
        
    chm.load_data(time_series, signal_series)
    
    if not all(chm.df):
        raise ValueError('df not initiated')
    else:
        if chm.df.empty:
            raise ValueError('df is empty')
        
    return chm
        
@pytest.fixture
def timestep(chm, time_series)-> float:
    timestep = chm.get_timestep(time_series)
    assert timestep, "timestep not initialized"
    
    assert timestep > 0, "timestep unrealistic"
    return timestep

def test_timestep(timestep, time_series, loaded_chm):
    assert timestep == loaded_chm.get_timestep(time_series), "calculated timesteps differ"
    
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