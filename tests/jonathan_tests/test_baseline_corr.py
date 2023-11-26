import numpy as np
import pytest
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import copy
from hplc_py.quant import Chromatogram
from hplc_py.hplc_py_typing.hplc_py_typing import checkArrayLike

"""
2023-11-27 06:26:22

test `correct_baseline`
"""

def test_get_tform(loaded_chm: Chromatogram, signal_array):
    
    tform = loaded_chm.baseline.compute_compressed_signal(signal_array)
    assert np.all(tform)
    assert isinstance(tform, np.ndarray)

@pytest.fixture
def windowsize():
    return 5

@pytest.fixture
def bcorr_col(intcol:str)->str:
    return intcol+"_corrected"
    
def test_correct_baseline(chm: Chromatogram,
                          time_array: ArrayLike,
                          timestep: int|float,
                          signal_array: ArrayLike,
                          windowsize
                          )->None:
    
    
    background_corrected_intensity, background = chm.baseline.correct_baseline(signal_array, windowsize, timestep)
    
    # pass the test if the area under the corrected signal is less than the area under the raw signal
    
    from scipy import integrate
    
    raw_auc = integrate.trapezoid(signal_array, time_array)
    
    bcorr_auc = integrate.trapezoid(background_corrected_intensity, time_array)
    
    assert raw_auc>bcorr_auc
    
    
    
    
    