from typing import Self
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes as Axes
from matplotlib.colors import ListedColormap
from numpy import float64
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series

from hplc_py.hplc_py_typing.hplc_py_typing import (
    PeakMap,
)
from hplc_py.hplc_py_typing.type_aliases import rgba
from hplc_py.io_validation import IOValid


@dataclass
class PeakMapViz(IOValid):
    """
    Provides a number of chainable plot functions to construct piece-wise a overlay plot of the properties mapped via MapPeaks.

    The ax the overlay is plotted on can be accessed through the `.ax` get method.
    """

    df: DataFrame
    x_col: str
    ax: Optional[Axes]=None

    def __post_init__(
        self,
    ):
        if not self.ax:
            fig = plt.figure()
            
            self.ax = fig.subplots() #type: ignore
        
        self._check_df(self.df)
        
        self._base_cmap: ListedColormap = mpl.colormaps["Set1"]
        self._cmap = self._base_cmap.resampled(len(self.df))
        
    def plot_peak_map(
        self
    ):
        self._plot_peaks()

    def _plot_peaks(
        self,
        pmax_col: str,
        label: Optional[str] = "y",
        plot_kwargs: dict = {},
    ) -> Self:
        
        """
        Plot peaks from the peak map, x will refer to the time axis, y to amp.
        """
        
        self._check_keys_in_index([self.x_col, pmax_col], self.df.columns)

        breakpoint()
        for i, s in self.df.iterrows():
            label_ = f"{label}_{i}"

            color: rgba = self._cmap.colors[i]  # type: ignore

            self.ax = self._plot_peak_factory(
                x=s[self.x_col],
                y=s[pmax_col],
                label=label_,
                color=color,
                ax=self.ax,
                plot_kwargs=plot_kwargs,
            )

        breakpoint()
        return self

    def _plot_peak_factory(
        self,
        x: NDArray[float64],
        y: NDArray[float64],
        label: str,
        color: rgba,
        ax: Axes,
        plot_kwargs: dict[str, Any] = {},
    ) -> Axes:
        """
        plot each peak individually.
        """
        ax.plot(
            x,
            y,
            label=label,
            c=color,
            marker="o",
            linestyle="",
            **plot_kwargs,
        )
        ax.legend()
        ax.plot()

        return ax

    def _plot_widths(
        self,
        df: DataFrame,
        wh_key: str,
        left_ips_key: str,
        right_ips_key: str,
        kind: Optional[Literal["line", "marker"]] = "marker",
        ax: Optional[Axes] = None,
        label: Optional[str] = "width",
        plot_kwargs: dict = {},
    ):
        """
        Main interface for plotting the widths. Wrap this in an outer function for specific measurements i.e. whh, bases
        """

        self._check_keys_in_index([left_ips_key, right_ips_key], df.columns)

        ls = ""

        if kind == "line":
            ls = "-"

        if not ax:
            ax = plt.gca()

        for idx, s in df.iterrows():
            color: rgba = self._base_cmap.colors[int(idx)]  # type: ignore

            ax = self._plot_width_factory(
                Series[float](s),
                idx,
                ls,
                wh_key,
                left_ips_key,
                right_ips_key,
                color,
                ax,
                plot_kwargs,
                label,
            )
        return ax

    def _plot_width_factory(
        self,
        s: Series[float],
        idx: Hashable,
        ls: Literal["", "-"],
        y_key: str,
        left_key: str,
        right_key: str,
        color: rgba,
        ax: Optional[Axes] = None,
        plot_kwargs: dict = {},
        label: Optional[str] = None,
    ):
        """
        For plotting an individual peaks widths measurements(?)

        Use timestep if signal x axis has a different unit to the calculated ips i.e. using units rather than idx
        """

        if isinstance(s, pd.Series):
            if s.empty:
                raise ValueError("series is empty")
        else:
            raise TypeError(f"df expected to be Dataframe, got {type(s)}")

        self._check_keys_in_index(
            [y_key, left_key, right_key],
            s.index,
        )

        color = self._base_cmap.colors[idx]  # type: ignore
        label = f"{label}_{idx}"

        s[[left_key, right_key]] = s[[left_key, right_key]]

        self._draw_widths(
            s,
            left_key,
            right_key,
            y_key,
            color,
            label,
            ls,
            ax,
            plot_kwargs,
        )

        return ax

    def _draw_widths(
        self,
        s: pd.Series,
        left_x_key: str,
        right_x_key: str,
        y_key: str,
        color: rgba,
        label: str,
        ls: Literal["", "-"],
        ax: Axes,
        plot_kwargs: dict = {},
    ):
        marker = "x"

        pkwargs = plot_kwargs.copy()

        if "marker" in pkwargs:
            marker = pkwargs.pop("marker")

        if ls == "-":
            marker = ""

        ax.plot(
            [s[left_x_key], s[right_x_key]],
            [s[y_key], s[y_key]],
            c=color,
            label=label,
            marker=marker,
            ls=ls,
            **pkwargs,
        )
        ax.legend()

        return ax

    def plot_whh(
        self,
        y_colname: str,
        left_colname: str,
        right_colname: str,
        ax: Optional[Axes] = None,
        kind: Optional[Literal["line", "marker"]] = "marker",
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Create whh plot specifically.
        """

        label = "whh"

        ax = self._plot_widths(
            self.df,
            y_colname,
            left_colname,
            right_colname,
            kind,
            ax,
            label,
            plot_kwargs,
        )

        return self

    def plot_bases(
        self,
        y_colname: str,
        left_colname: str,
        right_colname: str,
        ax: Optional[Axes] = None,
        kind: Optional[Literal["line", "marker"]] = "marker",
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Plot peak bases. 
        """

        label = "bases"
        
        ax = self._plot_widths(
            self.df,
            y_colname,
            left_colname,
            right_colname,
            kind,
            ax,
            label,
            plot_kwargs,
        )

        return self

