import pytest
import pandas as pd
import numpy as np
from hplc_py.hplc_py_typing.hplc_py_typing import checkArrayLike
from hplc_py.quant import Chromatogram

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
def time_array(testdata):
    assert isinstance(testdata, pd.DataFrame)
    assert checkArrayLike(testdata.x)
    return testdata.x.values

@pytest.fixture
def signal_array(testdata):
    assert checkArrayLike(testdata.y)
    return testdata.y.values

@pytest.fixture
def timecol():
    return 'time'

@pytest.fixture
def intcol():
    return 'signal'

@pytest.fixture
def loaded_chm(chm, time_array, signal_array)->Chromatogram:
    if not isinstance(chm, Chromatogram):
        raise TypeError("chm is not a Chromatogram object")
    
    if not checkArrayLike(time_array):
        raise TypeError("x must be ArrayLike")
    
    if not checkArrayLike(signal_array):
        assert TypeError(f"y must be ArrayLike, but passed {type(signal_array)}")
        
    chm.load_data(time_array, signal_array)
    
    if not all(chm.df):
        raise ValueError('df not initiated')
    else:
        if chm.df.empty:
            raise ValueError('df is empty')
        
    return chm

@pytest.fixture
def timestep(chm: Chromatogram, time_array)->float:
    timestep =  chm.compute_timestep(time_array)
    assert timestep
    assert isinstance(timestep, float)
    assert timestep>0
    
    return timestep