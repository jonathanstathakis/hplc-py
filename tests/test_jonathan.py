import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hplc_py import quant
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
    return quant.Chromatogram()

@pytest.fixture
def time_series(testdata):
    assert isinstance(testdata, pd.DataFrame)
    assert checkArrayLike(testdata.x)
    return testdata.x.values

@pytest.fixture
def signal_series(testdata):
    assert checkArrayLike(testdata.y)
    return testdata.y.values

def test_load_data(chm, time_series, signal_series)->None:
    
    if not isinstance(chm, quant.Chromatogram):
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