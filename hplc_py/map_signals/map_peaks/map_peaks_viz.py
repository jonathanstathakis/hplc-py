from typing import Any, Self

import distinctipy
import matplotlib.pyplot as plt
import pandas as pd
import pandera as pa
import polars as pl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy import float64
from numpy.typing import NDArray
from pandera.typing import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import PeakMap, X_Schema
from hplc_py.hplc_py_typing.type_aliases import rgba
from hplc_py.io_validation import IOValid
from hplc_py.show import SignalPlotter

from .map_peaks_viz_schemas import (
    Maxima_X_Y,
    PM_Width_In_X,
    PM_Width_In_Y,
    PM_Width_Long_Joined,
    PM_Width_Long_Out_X,
    PM_Width_Long_Out_Y,
    Width_Maxima_Join,
)

from .map_peaks_viz_pipelines import Pipeline_Peak_Map_Interface


class PlotCore(IOValid):
    """
    Base Class of the peak map plots, containing style settings and common methods.
    """

    def __init__(
        self,
        ax: Axes,
        df=pl.DataFrame(),
        colors: list[tuple[float, float, float]] = [
            (0.0, 0.0, 0.0),
        ],
    ):
        plt.style.use("ggplot")

        self.df = df
        self.ax = ax
        self.colors = colors

    def add_proxy_artist_to_legend(
        self,
        label: str,
        line_2d: Line2D,
    ) -> Self:
        handles, _ = self.ax.get_legend_handles_labels()
        legend_entry = self._set_legend_entry(label, line_2d)
        new_handles = handles + [legend_entry]
        self.ax.legend(handles=new_handles)

        return self

    def _set_legend_entry(self, label: str, line_2d: Line2D) -> Line2D:
        """
        Creates an empty Line2D object to use as the marker for the legend entry.

        Used to generate a representation of a marker category when there are too
        many entries for a generic legend.

        :param label: the legend entry text
        :type label: str
        :param line_2d: a representative line_2d of the plotted data to extract marker
        information from. Ensures that the legend marker matches what was plotted.
        """

        legend_marker = line_2d.get_marker()
        legend_markersize = line_2d.get_markersize()

        # TODO: modify this to a more appropriate method of choosing a 'representative'
        # color
        legend_color = self.colors[-1]
        legend_markeredgewidth = line_2d.get_markeredgewidth()
        legend_markeredgecolor = line_2d.get_markeredgecolor()

        legend_entry = Line2D(
            [],
            [],
            marker=legend_marker,
            markersize=legend_markersize,
            color=legend_color,
            markeredgewidth=legend_markeredgewidth,
            markeredgecolor=legend_markeredgecolor,
            label=label,
            ls="",
        )

        return legend_entry


class MaximaPlotter(PlotCore):

    @pa.check_types
    def __init__(
        self,
        ax: Axes,
        colors: list[tuple[float, float, float]],
        df=DataFrame[Maxima_X_Y],
    ):
        plt.style.use("ggplot")

        self.df = df
        self.ax = ax
        self.colors = colors
        
        self.x_colname = str(Maxima_X_Y.maxima_x)
        self.y_colname = str(Maxima_X_Y.maxima_y)

    def draw_maxima(
        self,
    ) -> Self:
        """
        Plot peaks from the peak map, x will refer to the time axis, y to amp.

        use `maxima_x_y` frame.
        """
        
        i: int
        d: dict
        for i, d in enumerate(self.df.iter_rows(named=True)):
            self.ax.plot(
                self.x_colname,
                self.y_colname,
                data=d,
                marker="o",
                c=self.colors[i],
                markeredgecolor="black",
                label="_",
            )

            self.ax.annotate(
                text=d['p_idx'],
                xy=(d[self.x_colname], d[self.y_colname]),
                ha="center",
                va="top",
                textcoords="offset pixels",
                xytext=(0, 40),
            )

        self.add_proxy_artist_to_legend(
            label='maxima',
            line_2d=self.ax.lines[-1]
                                        )
        return self


class WidthPlotter(PlotCore):
    """
    For handling all width plotting - WHH and bases. The core tenant is that we want to
    plot the widths peak by peak in such a way that it is clear which mark belongs to
    which peak, and which is left and which is right.

    Provides an option for plotting markers at the ips `plot_widths_vertices`, and
    lines connecting the ips to the maxima in `plot_widths_edges`
    """

    def __init__(
        self,
        ax: Axes,
        peak_map: pl.DataFrame,
        colors: list[tuple[float, float, float]] = [
            (0.0, 0.0, 0.0),
        ],
    ):
        super().__init__(ax=ax, df=peak_map, colors=colors)

    def plot_widths_vertices(
        self,
        y_colname: str,
        left_x_colname: str,
        right_x_colname: str,
        marker: str,
        label: str = "width",
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Main interface for plotting the width ips as points.
        """
        # shift the marker up enough to mark the location rather than occlude the signal

        # df_max = self.df[y_key].max()
        # self.df[y_key] = self.df[y_key] + df_max * 0.001

        # self.check_keys_in_index([left_x_colname, right_x_colname], self.df.columns)

        for idx, s in self.df.iter_rows():
            color: rgba = self.colors[idx]  # type: ignore

            self.ax.plot(
                [s[left_x_colname], s[right_x_colname]],
                [s[y_colname], s[y_colname]],
                c=color,
                marker=marker,
                markeredgecolor="black",
                ls="",
                label="_",
                **plot_kwargs,
            )

        self.add_proxy_artist_to_legend(label=label, line_2d=self.ax.lines[-1])

        return self

    def plot_widths_edges(
        self,
        width_maxima_join: DataFrame[Width_Maxima_Join],
    ):
        pass

        return self


class UI_PlotPeakMap:
    """
    2024-02-09 11:06:29

    Intended to be the user interface for producing a plot visualising the information
    obtained by `MapPeaks`.

    It will need to plot the signal, then overlay the peak maxima and ips at half height
    and peak base. There will be a bool option for plotting the signal and plotting the peak maxima, and a string|dict option for the ips. It will then create the ax object
    internally and provide a show method which will call plt.show()
    """

    def __init__(
        self,
        X: DataFrame[X_Schema],
        peak_map: DataFrame[PeakMap],
        ax: Axes = plt.gca(),
    ):
        """
        Provide an Axes object to draw on. if not provided, will draw on the current
        Axes
        """
        self.ax = ax
        self.X = X
        self.peak_map = peak_map
        self._peak_colors: list[tuple[float, float, float]] = distinctipy.get_colors(
            len(self.peak_map)
        )
        self.peak_map_interface = Pipeline_Peak_Map_Interface()

        self._signal_plotter = SignalPlotter(df=self.X, ax=self.ax)
        self._width_plotter = WidthPlotter(
            ax=ax, peak_map=peak_map, colors=self._peak_colors
        )

        # self._peak_plotter = PeakPlotter(ax=ax)

    def draw_signal(self) -> Self:
        """
        Call `SignalPlotter.plot_signal` to draw the input signal onto the internal Axes object.
        """
        self._signal_plotter.plot_signal(
            x_colname="X_idx",
            y_colname="X",
            label="X",
        )
        return self

    def draw_maxima(self) -> Self:
        """
        call `PeakPlotter.plot_maxima` to draw the maximas on the internal Axes. Needs to first subset the peak_map to `maxima_x_y`.

        :return: _description_
        :rtype: Self
        """
        
        maxima_x_y = (
            self.peak_map_interface._pipe_peak_maxima_to_long
        .load_pipeline(
            peak_map=self.peak_map,
            )
        .run_pipeline()
        .maxima_x_y
        )
        
        maxima_plotter = MaximaPlotter(
            ax=self.ax,
            df=maxima_x_y,
            colors=self._peak_colors
            )
        
        maxima_plotter.draw_maxima()
        
        return self
        

    def show(self) -> Self:
        """
        display the current Axes.

        Graphics(?) are drawn onto the internal Axes object by other methods within this class. Calling this method uses up the current Axes.
        """
        plt.legend()
        plt.show()

        return self


class PeakMapViz(IOValid):
    """
    Provides a number of chainable plot functions to construct piece-wise a overlay plot of the properties mapped via MapPeaks.

    The ax the overlay is plotted on can be accessed through the `.ax` get method.
    """

    def __init__(
        self,
        df: DataFrame,
        x_colname: str,
        ax: Axes = plt.gca(),
    ):
        self.df = df.copy(deep=True)
        self.x_colname = x_colname
        self.ax = ax
        self.pp = MaximaPlotter(ax, df.copy(deep=True))
        self.wp = WidthPlotter(ax)

    def plot_whh(
        self,
        y_colname: str,
        left_colname: str,
        right_colname: str,
        marker: str = "v",
        ax: Axes = plt.gca(),
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Create whh plot specifically.
        """

        label = "whh"

        wp = WidthPlotter(ax)

        wp.plot_widths_vertices(
            df=self.df,
            y_colname=y_colname,
            left_x_colname=left_colname,
            right_x_colname=right_colname,
            left_y_key="",
            right_y_key="",
            marker=marker,
            label=label,
            plot_kwargs=plot_kwargs,
        )

        return self

    def plot_bases(
        self,
        y_colname: str,
        left_colname: str,
        right_colname: str,
        ax: Axes,
        marker: str = "v",
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Plot peak bases.
        """

        label = "bases"
        wp = WidthPlotter(ax)

        wp.plot_widths_vertices(
            df=self.df,
            y_colname=y_colname,
            left_x_colname=left_colname,
            right_x_colname=right_colname,
            left_y_key="",
            right_y_key="",
            marker=marker,
            label=label,
            plot_kwargs=plot_kwargs,
        )

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
            c=color,
            marker="7",
            linestyle="",
            **plot_kwargs,
            alpha=0.5,
            markeredgecolor="black",
        )

        return ax
