import pytest
import pandas as pd
import numpy as np

import pandera.typing as pt
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from hplc_py.quant import Chromatogram
from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike, SignalDFIn
from scipy import integrate

"""
2023-11-27 06:26:22

test `correct_baseline`
"""
@pytest.fixture
def amp(testsignal):
  return testsignal.amp.to_numpy(dtype=np.float64)

# testdata: pt.DataFrame[SignalDF]

@pytest.fixture
def compressed_amp(chm: Chromatogram, amp: npt.NDArray[np.float64])-> npt.NDArray[np.float64]:
  
  # intensity raw compressed
  intensity_rc = chm.baseline.compute_compressed_signal(amp)
  
  assert intensity_rc
  
  return intensity_rc

def test_get_tform(chm: Chromatogram,
                   amp
                   ):
    
    tform = chm.baseline.compute_compressed_signal(amp)
    
    assert np.all(tform)
    assert isinstance(tform, np.ndarray)


def test_compute_inv_tfrom(chm: Chromatogram,
                           amp,
                           )->None:

  chm.baseline.compute_inv_tform(amp)


@pytest.fixture
def bcorr(chm: Chromatogram, amp: npt.NDArray[np.float64], windowsize:int, timestep: np.float64):
  
  bcorr = chm.baseline.correct_baseline(amp, windowsize, timestep)[0]
  
  return bcorr
  
def test_correct_baseline(
                        amp,
                        bcorr,
                        time,
                          )->None:
    
    # pass the test if the area under the corrected signal is less than the area under the raw signal
    
    raw_auc = integrate.trapezoid(amp, time)
    bcorr_auc = integrate.trapezoid(bcorr, time)
    
    assert raw_auc>bcorr_auc
    

