import pandera as pa
from typing import Type
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import pandas as pd
from hplc_py.io_validation import IOValid
import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray
from hplc_py.misc.misc import compute_timestep
from .hplc_py_typing.hplc_py_typing import SignalDFLoaded, PeakMap, Data
from .show import Show, PlotSignal
import polars as pl
from typing import Self
from pandera.typing import Series, DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import WindowedTime

class Chromatogram(IOValid):
    """
    The class representing a Chromatogram, a 2D signal of amplitude (y) over time (x). It acts as the data repository of the pipeline, collecting the data as it progresses through the pipeline. Validation will be performed here.
    
    The chromatogram should draw itself onto a given canvas, i.e. it will own the plotting functions, which are called by the pipeline.
    """
    
    def __init__(
        self,
        time: NDArray[float64],
        amp: NDArray[float64],
    ):
        """
        validate the input arrays  and use to intialize the internal data attribute
        """
        
        self._sch_data: Type[Data] = Data
        self._sch_peakmap: Type[Data] = PeakMap
        self.bcorr_corrected = False
        self.__sigld_sch = SignalDFLoaded

                
        self.timestep = compute_timestep(time)
        self.load_df(time, amp)
    
    def load_df(
        self,
        time,
        amp,
    ):
        """
        validate the input time and amp arrays, construct a frame, validate it against the DFLoaded schema, then add as 'data'
        """
        for n, s in {"time": time, "amp": amp}.items():
            self.check_container_is_type(s, np.ndarray, float64, n)
            
        data_ = pd.DataFrame({self.__sigld_sch.time: time, self.__sigld_sch.amp: amp}).reset_index(names=self.__sigld_sch.time_idx).rename_axis(index=self.__sigld_sch.idx)

        Data.validate(data_, lazy=True)
        
        self._data = DataFrame[Data](data_)
    
    @property
    def df_pl(
        self,
    ):
        return pl.from_pandas(self._data)
    
    @df_pl.setter
    @pa.check_types
    def join_data_to_windowed_time(
        self,
        windowed_time: DataFrame[WindowedTime],
    ):
    
        windowed_time = pl.from_pandas(windowed_time)
            
        df = self.df_pl.join(windowed_time, how='left', on='time_idx').to_pandas()
        
        self.df = df
        
        
        breakpoint()
    
    @property
    def df_pd(
        self,
    ):
        return self._data

    @property
    def amp(
        self
    ):
        """
        Returns different column from data depending on whether 'bcorr_corrected' flag is True or False.
        If True, returns 'amp_corrected', else returns 'amp'.
        """
        if self.bcorr_corrected:
            return self._data['amp_corrected']
        else:
            return self._data['amp']
        
    @property
    def peakmap(
        self,
    ):
        return self._peakmap
    
    @peakmap.setter
    def peakmap(
        self,
        value
    ):
        PeakMap.validate(value, lazy=True)
        self._peakmap = value
    
    
    @property
    def time(
        self
    ):
        return self._data['time']
    
    @property
    def ws(
        self,
    ):
        return self._ws
        
    
    @ws.getter
    def ws(
        self
    ):
        """
        Return columns from the central data object depending on presence.
        
        If "amp_corrected" is in `data`, return that as the 'amp'.
        if "amp_unmixed" is in `data`, return that as well.
        """ 
        
        ws_cols = ['w_type','w_idx','time_idx','time']
        if "amp_corrected" in self._data.columns:
            ws_cols.append('amp_corrected')
        else:
            ws_cols.append("amp")
        
        if "amp_unmixed" in self._data.columns:
            ws_cols.append("amp_unmixed")
            
        return self._data[ws_cols]
        
    @ws.setter
    def ws(
        self,
        value
    ):
        self._ws = value
    
    @property
    def _data_idx_cols(
        self
    ):
        """
        A list of columns expected to be in Data, but possibly not present. Ideally this
        would come from the schema but not sure how to implement ATM.
        """
        
        return ['w_type','w_idx','time_idx','time']
    
    @property
    def _data_value_cols(
        self
    ):
        """
        A list of columns expected to be in Data but possibly not present for a given state.
        Ideally this would come frmo the schema but not sure how to implement ATM.
        """
        
        return ['amp','amp_corrected','amp_unmixed']
    
    @property
    def _present_x_cols(
        self
    ):
        """
        Compare the `_data_idx_cols` with the current frame to identify which are actually present for plotting purposes
        """
        
        return self.df_pd.columns[self.df_pd.columns.isin(self._data_idx_cols)]

        
    @property
    def _present_y_cols(
        self
    ):
        """
        Only provide y column names that are currently present in the data as per the
        data Schema. Use for plotting
        """
        
        return list(self.df_pd.columns[self.df_pd.columns.isin(self._data_value_cols)])
        
        
    def plot_signals(
        self,
    )->Self:
        
        x = self._sch_data.time
        
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.subplots()
        
        for y in self._present_y_cols:
            
            ax.plot(x, y, data=self.df_pl, label=self.df_pl[y].name)
            
        plt.legend()
        plt.show()
        
        return self
        
    def __repr__(
        self
    ):
        return (f"DATA:\n{pl.from_pandas(self._data).__repr__()}\n"
                "\n"
                f"TIMESTEP:\n{self.timestep}\n"
                
                )