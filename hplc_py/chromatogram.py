from dataclasses import dataclass, field
import pandas as pd
from hplc_py.io_validation import IOValid
import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray
from hplc_py.misc.misc import compute_timestep
from .hplc_py_typing.hplc_py_typing import SignalDFLoaded
from .show import Show, PlotSignal
import polars as pl
from typing import Self
from pandera.typing import Series, DataFrame

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
        self.__sigld_sch = SignalDFLoaded

        for n, s in {"time": time, "amp": amp}.items():
            self.check_container_is_type(s, np.ndarray, float64, n)

        self.data = DataFrame[SignalDFLoaded](
                pd.DataFrame({self.__sigld_sch.time: time, self.__sigld_sch.amp: amp}
                             )
                .reset_index(names=self.__sigld_sch.time_idx)
                .rename_axis(index=self.__sigld_sch.idx)
        )

        self.timestep = compute_timestep(time)
    
    def plot_signal(
        self,
        ax,
    )->Self:
        
        PlotSignal(self.data,
                   str(self.__sigld_sch.time),
                   str(self.__sigld_sch.amp),
                   ax=ax,
                   )._plot_signal_factory()
        
        return self
    
    
        
        
    def __repr__(
        self
    ):
        return (f"DATA:\n{pl.from_pandas(self.data).__repr__()}\n"
                "\n"
                f"TIMESTEP:\n{self.timestep}\n"
                
                )
