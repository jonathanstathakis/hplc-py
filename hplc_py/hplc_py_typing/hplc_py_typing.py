from typing import Any

import pandera as pa
import pandera.typing as pt

import numpy.typing as npt
import numpy as np

class SignalDFIn(pa.DataFrameModel):
    """
    The base signal, with time and amplitude directions
    """
    
    time: pt.Series[np.float64]
    amp: pt.Series[np.float64]

class WindowedSignalDF(pa.DataFrameModel):
    """
    Contains a recording of each window in the Chromatogram labeled with an ID and type
    with a time index corresponding to the time values in the time array.
    
    Spans the length of the chromatogram signal
    """
    time_idx: pt.Series[np.int64]
    time: pt.Series[np.float64]
    amp: pt.Series[np.float64]
    window_id: pt.Series[np.int64]
    window_type: pt.Series[str] # either 'peak' or 'np.int64erpeak'
    
class WidthDF(pa.DataFrameModel):
    """
    Contains information about the dimensions of each peak, labeled with 'peak_idx', the
    idx location of the peak maxima in the np.int64ensity array, the width of the peak in
    index units, 'clh' is the contour line height of the peak where the width was measured,
    'left' is the index value of the left bound of the peak, 'right' corresponds to the
    right.
    """
    peak_idx: pt.Series[np.int64]
    width: pt.Series[np.float64]
    clh: pt.Series[np.float64]
    left: pt.Series[np.float64]
    right: pt.Series[np.float64]
    
class PeakWindowDF(pa.DataFrameModel):
    
    peak_idx: pt.Series[np.int64]
    time: pt.Series[np.float64]
    signal: pt.Series[np.float64]

def isArrayLike(x: Any):
    
    if not any(x):
        raise ValueError("x is None")
    
    if not hasattr(x, "__array__"):
        return False
    else:
        return True