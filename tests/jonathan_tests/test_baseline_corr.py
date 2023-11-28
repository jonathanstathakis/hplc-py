import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from hplc_py.quant import Chromatogram
from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike
from scipy import integrate

"""
2023-11-27 06:26:22

test `correct_baseline`
"""

def test_get_tform(loaded_chm: Chromatogram, intensity_raw):
    
    tform = loaded_chm.baseline.compute_compressed_signal(intensity_raw)
    
    assert np.all(tform)
    assert isinstance(tform, np.ndarray)

@pytest.fixture
def compressed_intensity(chm: Chromatogram, intensity_raw: npt.NDArray[np.float64])-> npt.NDArray:
  
  # intensity raw compressed
  intensity_rc = chm.baseline.compute_compressed_signal(intensity_raw)
  
  print(type(intensity_rc))


def test_compute_inv_tfrom(chm: Chromatogram, intensity_raw: npt.NDArray[np.float64])->None:
  # chm.baseline.
  chm.baseline.compute_inv_tform(intensity_raw)

def test_correct_baseline(
                        intensity_raw,
                        time,
                        intensity_corrected,
                          )->None:
    
    
    
    # pass the test if the area under the corrected signal is less than the area under the raw signal
    
    raw_auc = integrate.trapezoid(intensity_raw, time)
    bcorr_auc = integrate.trapezoid(intensity_corrected, time)
    
    assert raw_auc>bcorr_auc
    

