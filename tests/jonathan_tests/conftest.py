import pytest
import pandas as pd
import pandera as pa
import pandera.typing as pt
import numpy as np
import numpy.typing as npt

from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike, SignalDFIn
from hplc_py.quant import Chromatogram

class TestDataDF(pa.DataFrameModel):
    x: pt.Series[float]
    y: pt.Series[float]
              
@pytest.fixture
def testdata_path():
    return "tests/test_data/test_many_peaks.csv"

@pytest.fixture
def testsignal(testdata_path)->pt.DataFrame[TestDataDF]:
    data = pd.read_csv(testdata_path)
    
    assert isinstance(data, pd.DataFrame)
    
    data = data.rename({
        'x':'time',
        'y':'amp'
    }, axis=1
                                  )
    
    return data.pipe(pt.DataFrame[SignalDFIn])
    
@pytest.fixture
def chm():
    return Chromatogram(viz=False)

@pytest.fixture
def time(testsignal: pt.DataFrame[TestDataDF]):
    assert isinstance(testsignal, pd.DataFrame)
    assert isArrayLike(testsignal.time)
    return testsignal.time.values

@pytest.fixture
def timestep(chm: Chromatogram, time:npt.NDArray[np.float64])->float:
    timestep =  chm.compute_timestep(time)
    assert timestep
    assert isinstance(timestep, float)
    assert timestep>0
    
    return timestep

@pytest.fixture
def amp(testsignal):
    assert isArrayLike(testsignal.amp)
    return testsignal.amp.values

@pytest.fixture
def windowsize():
    return 5

@pytest.fixture
def bcorr_col(intcol:str)->str:
    return intcol+"_corrected"

@pytest.fixture
def timestep(chm: Chromatogram, time: npt.NDArray[np.float64])->np.float64:
  timestep = chm.compute_timestep(time)
  return timestep

@pytest.fixture
def bcorr(chm: Chromatogram, amp: npt.NDArray[np.float64], windowsize:int, timestep: np.float64):
  
  bcorr = chm.baseline.correct_baseline(amp, windowsize, timestep)[0]
  
  return bcorr

@pytest.fixture
def timecol():
    return 'time'

@pytest.fixture
def intcol():
    return 'signal'