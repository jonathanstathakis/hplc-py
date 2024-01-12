from typing import Any
from dataclasses import dataclass, field
import numpy as np
import pandera.typing as pt
import pandas as pd
import pandera as pa
from pandera.typing import Series

from hplc_py.hplc_py_typing.hplc_py_typing import SignalDFInBase, OutSignalDF_Base

@dataclass
class DataMixin:
    def _check_df_loaded(
        self,
        df: Any,
    ):
      
        if isinstance(df, pd.DataFrame):
            if df.empty:
                raise ValueError(f"frame is empty")
        else:
            raise TypeError(f"input must be a dataframe, got {type(df)}")
        
    
class LoadData:
    # internal signal table
    time_col: str = 'time'
    amp_col: str = 'amp_raw'
    _signal_df: pt.DataFrame[SignalDFInBase] = field(default=None)
    _df_alias: str = '`_signal_df`'
    
        
    @pa.check_types
    def set_signal_df(
                self,
                signal_df: pd.DataFrame,
                ):
            
        """
        Load the data, providing the keys to the time and amplitude columns.
        
        signal_df: a dataframe consisting of an amplitude column indexed by a time column.
        time_window: ..
        
        returns the loaded dataframe for user inspection
        """
        
        # input validation
   
        if not isinstance(signal_df, pd.DataFrame):
            raise TypeError("signal_df must be a pandas dataframe")
        
        if not signal_df.columns.isin([self.time_col, self.amp_col]).any():
            raise ValueError("supplied keys not in dataframe columns.\nplease label time column as 'time' and amp column as 'amp'")
        
        # enforce only the time and amp col in the dataframe.
        
        if len(signal_df.columns.drop([self.time_col, self.amp_col])) > 0:
            raise ValueError(f"Please input a DataFrame with only the time and amp columns.")

        # store the chromatogram df and (re)name the column and index
        
        signal_df = pd.DataFrame({
            self.time_col:signal_df[self.time_col],
            self.amp_col:signal_df[self.amp_col],
        }).astype({
            self.time_col: pd.Float64Dtype(),
            self.amp_col: pd.Float64Dtype(),
        })
        
        signal_df = signal_df.reset_index(names='time_idx')
        
        self._signal_df = signal_df
        
        return pt.DataFrame[OutSignalDF_Base](self._signal_df)
    

@dataclass
class TimeStep:
    
    _timestep: float = None
    
    def set_timestep(
        self,
        time: Series[float],
    )->None:
        self._timestep = self.compute_timestep(time)
    
    def compute_timestep(self, time_array: Series[np.float64])->np.float64:
        # Define the average timestep in the chromatogram. This computes a mean
        # but values will typically be identical.
        
        dt = np.diff(time_array)
        mean_dt = np.mean(dt)
        return mean_dt.astype(np.float64)
    
