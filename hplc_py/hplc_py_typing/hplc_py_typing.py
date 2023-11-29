from typing import Any

import pandas as pd
import pandera as pa
import pandera.typing as pt

import numpy.typing as npt
import numpy as np

class SignalDFIn(pa.DataFrameModel):
    """
    The base signal, with time and amplitude directions
    """
    
    time: np.float64
    amp: np.float64

class BaseWindowedSignalDF(pa.DataFrameModel):
    """
    Contains a recording of each window in the Chromatogram labeled with an ID and type
    with a time index corresponding to the time values in the time array.
    
    Spans the length of the chromatogram signal
    """

    time_idx: pd.Int64Dtype=pa.Field(coerce=False)
    time: pd.Float64Dtype=pa.Field(coerce=False)
    amp: pd.Float64Dtype=pa.Field(coerce=False)
    norm_amp: pd.Float64Dtype=pa.Field(coerce=False)
    window_id: pd.Int64Dtype=pa.Field(nullable=True, coerce=False)
    range_idx: pd.Int64Dtype=pa.Field(nullable=True, coerce=False)
    window_type: str # either 'peak' or 'np.int64erpeak'

class TestOutWindowedSignalDF(BaseWindowedSignalDF):
    time_idx: pd.Int64Dtype = pa.Field(
        nullable=True,
        ge=0,
        lt=7000
        )
    time: pd.Float64Dtype =  pa.Field(
        nullable=True,
        ge=0,
        lt=70
        )
    amp: pd.Float64Dtype = pa.Field(
        nullable=True,
        ge=6.425446e-48,
        le=3.989453e+01
        )
    norm_amp: pd.Float64Dtype = pa.Field(
        nullable=True,
        ge=0,
        le=1
        )
    window_id: pd.Int64Dtype= pa.Field(
        nullable=True,
        isin=[0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
    window_type: str=pa.Field(
        nullable=True,
        isin=['peak', 'interpeak']
        )

class PeakDF(pa.DataFrameModel):
    """
    Contains information about each detected peak, used for profiling
    """
    
    maxima_idx: pd.Int64Dtype=pa.Field(coerce=False) # the time idx values corresponding to the peak maxima location
    peak_prom: pd.Float64Dtype=pa.Field(coerce=False)
    whh: pd.Float64Dtype=pa.Field(coerce=False)
    whhh: pd.Int64Dtype=pa.Field(coerce=False)
    whh_left: pd.Int64Dtype=pa.Field(coerce=False)
    whh_right: pd.Int64Dtype=pa.Field(coerce=False)
    rl_width: pd.Float64Dtype=pa.Field(coerce=False)
    rl_wh: pd.Int64Dtype=pa.Field(coerce=False)
    rl_left: pd.Int64Dtype=pa.Field(coerce=False)
    rl_right: pd.Int64Dtype=pa.Field(coerce=False)
    
    @pa.dataframe_check
    def check_null(cls, df: pd.DataFrame)->bool:
        return df.shape[0]>0

class TestOutPeakDF(PeakDF):
    
    maxima_idx: pd.Int64Dtype=pa.Field(
        nullable=True,
        isin=[995, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5505]
        )
    peak_prom: pd.Float64Dtype=pa.Field(
        nullable=True,
        isin=[0.972496399228961, 1.0]
        )
    
    whh: pd.Float64Dtype=pa.Field(
        nullable=True,
        isin=[166.0, 203.0, 167.0, 191.0]
        )
    whhh: pd.Int64Dtype=pa.Field(
        nullable=True,
        isin=[0]
        )
    whh_left: pd.Int64Dtype=pa.Field(
        nullable=True,
        isin=[890, 1415, 1916, 2416, 2916, 3416, 3916, 4416, 4916, 5418]
        )
    whh_right: pd.Int64Dtype=pa.Field(
        nullable=True,
        isin=[1093, 1583, 2083, 2583, 3083, 3583, 4083, 4583, 5083, 5609]
        )
    
    rl_width: pd.Float64Dtype=pa.Field(
        nullable=True,
        isin =[166.0, 203.0, 167.0, 191.0]
        )
    rl_wh: pd.Int64Dtype=pa.Field(
        nullable=True,
        isin =[0]
        )
    rl_left: pd.Int64Dtype=pa.Field(
        nullable=True,
        isin =[890, 1415, 1916, 2416, 2916, 3416, 3916, 4416, 4916, 5418]
        )
    rl_right: pd.Int64Dtype=pa.Field(
        nullable=True,
        isin=[1093, 1583, 2083, 2583, 3083, 3583, 4083, 4583, 5083, 5609]
        )

def isArrayLike(x: Any):
    
    if not any(x):
        raise ValueError("x is None")
    
    if not hasattr(x, "__array__"):
        return False
    else:
        return True
    

class BaseAugmentedDF(BaseWindowedSignalDF):
    """
    Combination of PeakDF and BaseWindowedSignalDF but with nullable fields for the peak properties
    """
    
    maxima_idx: pd.Int64Dtype=pa.Field(nullable=True) # the time idx values corresponding to the peak maxima location
    peak_prom: pd.Float64Dtype=pa.Field(nullable=True)
    whh: pd.Float64Dtype=pa.Field(nullable=True)
    whhh: pd.Int64Dtype=pa.Field(nullable=True)
    whh_left: pd.Int64Dtype=pa.Field(nullable=True)
    whh_right: pd.Int64Dtype=pa.Field(nullable=True)
    rl_width: pd.Float64Dtype=pa.Field(nullable=True)
    rl_wh: pd.Int64Dtype=pa.Field(nullable=True)
    rl_left: pd.Int64Dtype=pa.Field(nullable=True)
    rl_right: pd.Int64Dtype=pa.Field(nullable=True)
    
    pass

class AugmentedDataFrameWidthMetrics(BaseAugmentedDF):
    """
    The final dataframe output
    """
    window_area: pd.Float64Dtype=pa.Field(nullable=True)
    num_peaks: pd.Int64Dtype=pa.Field(nullable=True)
    
class TestAugmentedFrameWidthMetrics(
                            TestOutPeakDF,
                            TestOutWindowedSignalDF,
                            ):
        """
        Combination of WindowedSignalDF and PeakDF but fields are nullable.
        
        note: would inherit TestOutPeakDF but need to set field to nullable, dont know how to do that inheritance currently
        """
        
        window_area: pd.Float64Dtype=pa.Field(
            nullable=True,
            isin=[38825.51718537447, 5962.918997523824, 5962.918997523825, 6879.622566252637, 6563.702515498473, 5990.7247502076625]
            )
        num_peaks: pd.Int64Dtype=pa.Field(nullable=True, isin=[0,1])
        amplitude: pd.Float64Dtype=pa.Field(nullable=True, isin=[28.27613266537977, 39.89452538404621, 39.89437671209474])