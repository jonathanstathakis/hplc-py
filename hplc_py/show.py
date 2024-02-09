import matplotlib as mpl
from hplc_py.hplc_py_typing.hplc_py_typing import Data
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pandera as pa
import polars as pl
from matplotlib.axes import Axes as Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from numpy import float64
from pandera.typing import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import (
    WindowedSignal,
)

from hplc_py.io_validation import IOValid

"""
Module for vizualisation, primarily the "Show" class
"""

class SignalPlotter(IOValid):
    def __init__(
        self,
        df: pl.DataFrame,
        ax: Axes=plt.gca(),
    ):
        # self.check_df_is_pd_not_empty(df)
        self.ax = ax
        self.df = df
        

    def plot_signal(
        self,
        x_colname: str,
        y_colname: str,
        label: Optional[str] = None,
        line_kwargs: dict = {},
    ) -> None:

        self.check_keys_in_index([x_colname, y_colname], self.df.columns)
        sig_x = self.df[x_colname]
        sig_y = self.df[y_colname]

        line = self.ax.plot(sig_x, sig_y, label=label, **line_kwargs)
        self.ax.legend(handles=line)
        

class DrawPeakWindows:
    def __init__(self):
        self._sch_data: Type[WindowedSignal] = WindowedSignal

    @pa.check_types
    def draw_peak_windows(
        self,
        ws: DataFrame[WindowedSignal],
        ax: Axes,
        vspan_kwargs: dict = {},
    ):
        """
        Plot each window as a Rectangle

        height is the maxima of the signal.
        
        """
        
        ws_ = pl.from_pandas(ws)
        
        if not isinstance(ws_, pl.DataFrame):
            raise TypeError("expected ws to be a polars dataframe")
        
        window_bounds = self._find_window_bounds(ws_)
        
        grpby_obj = window_bounds.filter(pl.col("w_type") == "peak").group_by([
            str(self._sch_data.w_idx)], maintain_order=True
        )
        

        cmap = mpl.colormaps["Set1"].resampled(len(window_bounds))
        
        # handles, labels = ax.get_legend_handles_labels()

        for label, grp in grpby_obj:
            x0 = grp.item(0, "start")
            x1 = grp.item(0, "end")
            
            # testing axvspan
            
            ax.axvspan(x0, x1, label=f"peak window {label}", color = cmap.colors[label], alpha=0.25, **vspan_kwargs)


    def _find_window_bounds(self, ws_):
        
        window_bounds = (
            ws_.group_by([self._sch_data.w_type, self._sch_data.w_idx])
            .agg(
                start=pl.col(str(self._sch_data.time)).first(),
                end=pl.col(str(self._sch_data.time)).last(),
            )
            .sort("start")
        )
        
        return window_bounds
        
        

class Show(
):
    def __init__(self):
        pass

    def plot_signal(
        self,
        signal_df: pd.DataFrame,
        time_col: str,
        amp_col: str,
        ax: plt.Axes,
    ):
        x = signal_df[time_col]
        y = signal_df[amp_col]

        ax.plot(x, y, label="bc chromatogram")
        return ax

    def plot_reconstructed_signal(
        self,
        unmixed_df,
        ax,
    ):
        """
        Plot the reconstructed signal as the sum of the deconvolved peak series
        """
        amp_unmixed = (
            unmixed_df.pivot_table(columns="p_idx", values="amp_unmixed", index="time")
            .sum(axis=1)
            .reset_index()
            .rename({0: "amp_unmixed"}, axis=1)
        )
        x = amp_unmixed["time"]
        y = amp_unmixed["amp_unmixed"]

        ax.plot(x, y, label="reconstructed signal")

        return ax
