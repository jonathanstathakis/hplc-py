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
    PeakMap,
    WindowedSignal,
)
from hplc_py.hplc_py_typing.type_aliases import rgba

from hplc_py.io_validation import IOValid
from hplc_py.map_signals.map_peaks.map_peaks_viz import PeakMapViz

"""
Module for vizualisation, primarily the "Show" class
"""


@dataclass
class PlotSignal(IOValid):
    df: pd.DataFrame
    x_colname: str
    y_colname: str
    label: Optional[str] = None
    ax: Axes = plt.axes()
    line_kwargs: dict = field(default_factory=dict)
    
    def __post_init__(
        self,
    ):
        if not self.label:
            self.label = self.y_colname

    def _plot_signal_factory(
        self,
    ) -> Axes:

        self._check_df(self.df)
        self._check_keys_in_index([self.x_colname, self.y_colname], self.df.columns)

        sig_x = self.df[self.x_colname]
        sig_y = self.df[self.y_colname]

        self.ax.plot(sig_x, sig_y, label=self.label, **self.line_kwargs)
        self.ax.legend()

        return self.ax



@dataclass
class MapWindowPlots:
    ws_sch: Type[WindowedSignal] = WindowedSignal

    def __post_init__(self):
        super().__init__()

    def _rectangle_factory(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        angle: float = 0.0,
        rotation_point: Literal["xy"] = "xy",
        rectangle_kwargs={},
    ) -> Rectangle:
        rectangle = Rectangle(
            xy,
            width,
            height,
            angle=angle,
            rotation_point=rotation_point,
            **rectangle_kwargs,
        )

        return rectangle


@pa.check_types
def plot_windows(
    self,
    ws: DataFrame[WindowedSignal],
    height: float,
    ax: Optional[Axes] = None,
    rectangle_kwargs: dict = {},
):
    """
    Plot each window as a Rectangle

    height is the maxima of the signal.
    """
    ws_ = pl.from_pandas(ws)
    if not isinstance(ws_, pl.DataFrame):
        raise TypeError("expected ws to be a polars dataframe")

    if not ax:
        ax = plt.gca()

    # rectangle definition: class matplotlib.patches.Rectangle(xy, width, height, *, angle=0.0, rotation_point='xy', **kwargs)
    # rectangle usage: `ax.add_collection([Rectangles..])` or `ax.add_patch(Rectangle)``

    window_stats = ws_.group_by([self.ws_sch.w_type, self.ws_sch.w_idx]).agg(
        start=pl.col(str(self.ws_sch.time)).first(),
        end=pl.col(str(self.ws_sch.time)).last(),
    )

    rh = height * 1.05

    # peak windows
    rectangles = []
    for k, g in window_stats.group_by([self.ws_sch.w_type, self.ws_sch.w_idx]):
        x0 = g.item(0, "start")
        y0 = 0
        width = g.item(0, "end") - x0

        rectangle = self._rectangle_factory(
            (x0, y0),
            width,
            rh,
            rectangle_kwargs=rectangle_kwargs,
        )

        rectangles.append(rectangle)

    import matplotlib as mpl

    cmap = mpl.colormaps["Set1"].resampled(len(window_stats))

    pc = PatchCollection(
        rectangles,
        zorder=0,
        alpha=0.25,
        facecolors=cmap.colors,
    )
    ax.add_collection(pc)

    pass


class Show(
    PeakMapViz,
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
