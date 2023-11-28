import pytest
import pandas as pd
import pandera as pa
import pandera.typing as pt
import numpy as np
import numpy.typing as npt

from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike
from hplc_py.quant import Chromatogram

class TestDataDF(pa.DataFrameModel):
    x: pt.Series[float]
    y: pt.Series[float]
              
@pytest.fixture
def testdata_path():
    return "tests/test_data/test_many_peaks.csv"

@pytest.fixture
def testdata(testdata_path)->pt.DataFrame[TestDataDF]:
    data = pd.read_csv(testdata_path)
    
    assert isinstance(data, pd.DataFrame)
    
    return data
    
@pytest.fixture
def chm():
    return Chromatogram(viz=False)

@pytest.fixture
def time(testdata: pt.DataFrame[TestDataDF]):
    assert isinstance(testdata, pd.DataFrame)
    assert isArrayLike(testdata.x)
    return testdata.x.values

@pytest.fixture
def timestep(chm: Chromatogram, time:npt.NDArray[np.float64])->float:
    timestep =  chm.compute_timestep(time)
    assert timestep
    assert isinstance(timestep, float)
    assert timestep>0
    
    return timestep

@pytest.fixture
def intensity_raw(testdata):
    assert isArrayLike(testdata.y)
    return testdata.y.values

@pytest.fixture
def windowsize():
    return 5

@pytest.fixture
def bcorr_col(intcol:str)->str:
    return intcol+"_corrected"

@pytest.fixture
def intensity_corrected(chm: Chromatogram, timestep: float, intensity_raw: pt.Series[float], windowsize: int)->pt.Series:
    background_corrected_intensity = chm.baseline.correct_baseline(intensity_raw, windowsize, timestep)[0]
    return background_corrected_intensity

def background(chm: Chromatogram, timestep: float, signal_array: pt.Series[float], windowsize: int)->pt.Series[float]:
    background = chm.baseline.correct_baseline(signal_array, windowsize, timestep)[1]
    return background

@pytest.fixture
def timecol():
    return 'time'

@pytest.fixture
def intcol():
    return 'signal'

@pytest.fixture
def loaded_chm(chm, time, intensity_raw)->Chromatogram:
    if not isinstance(chm, Chromatogram):
        raise TypeError("chm is not a Chromatogram object")
    
    if not isArrayLike(time):
        raise TypeError("x must be ArrayLike")
    
    if not isArrayLike(intensity_raw):
        assert TypeError(f"y must be ArrayLike, but passed {type(intensity_raw)}")
        
    chm.load_data(time, intensity_raw)
    
    if not all(chm.df):
        raise ValueError('df not initiated')
    else:
        if chm.df.empty:
            raise ValueError('df is empty')
        
    return chm