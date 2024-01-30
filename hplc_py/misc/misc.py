from typing import Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numpy import float64

@dataclass
class DataMixin:
    def _check_df_loaded(
        self,
        df: Any,
    ):
      
        if isinstance(df, pd.DataFrame):
            if df.empty:
                raise ValueError("frame is empty")
        else:
            raise TypeError(f"input must be a dataframe, got {type(df)}")    
    
def compute_timestep(time_array: NDArray[float64])-> float64:
    # Define the average timestep in the chromatogram. This computes a mean
    # but values will typically be identical.
    
    dt = np.diff(time_array)
    mean_dt = np.mean(dt)
    return mean_dt.astype(float64)

