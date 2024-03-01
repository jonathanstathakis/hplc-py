"""
Pure functions for plotting information from `peak_map`.
"""

import holoviews as hv
import warnings
import numpy as npp
import hvplot
from matplotlib.colors import Colormap
import seaborn as sns
import colorcet as cc
import pandera as pa
import polars as pl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from pandera.typing import DataFrame
from hplc_py.map_peaks.schemas import PeakMapOutput

from hplc_py.map_peaks import definitions as mp_defs
from hplc_py.map_peaks.schemas import PeakMap
from typeguard import typechecked

from hplc_py.hplc_py_typing.hplc_py_typing import ColorMap
from typing import Self
from hplc_py.common.common_schemas import X_Schema

import matplotlib.pyplot as plt

from hplc_py.map_peaks.schemas import PeakMapWideColored

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class PeakMapPlotHandler:
    signal: Any
    maxima: Any
    whh: Any
    bases: Any

    def overlay(self, which: list[str] = ["all"]) -> Any:
        # TODO: correct return type
        # TODO: correct all types, use a literal list to provide 'which' input options
        # TODO: add options for 'which' in function body

        """
        Call to overlay valid internal peak objects. defaults to all which are not none.
        """

        not_nulls = [v for v in self if v is not None]

        peak_map_overlay = hv.Overlay(not_nulls)
        """
        TODO:
        modify to return the overlay plot 
        """

        return peak_map_overlay

    def __iter__(self):

        for field in fields(self):
            yield getattr(self, field.name)


class VizPeakMapFactory:
    """
    Peak Map Vizualisation. Intended to be inherited by MapPeaks, and its methods called through that subclass.

    Provides methods for plotting the maxima as a scatter, the WHH as scatter, and the bases as lines drawn from the maxima of the peak down to each base. Each method will provide a hvplot `holoviews.core.overlay.NDOverlay` object which will allow for downstream plot composition.
    """

    def __init__(
        self,
        X: DataFrame[X_Schema],
        peak_map: PeakMapOutput,
        ax: Axes = plt.gca(),
    ):
        """
        Provide an Axes object to draw on. if not provided, will draw on the current
        Axes
        """
        self.ax = ax
        self._X = X
        self.peak_map = peak_map

        peak_idx = (
            self.peak_map.maxima.pipe(pl.from_pandas)
            .select("p_idx")
            .unique(maintain_order=True)
        )

        self.peak_color_map = assign_colors_to_p_idx(p_idx=peak_idx)

        # self.peak_map = join_peak_map_colors(peak_map, assign_colors_to_p_idx(peak_map))

        self.maxima_idx_key = "X_idx"
        self.maxima_key = "maxima"
        self.color_key = "color"

        self.handles: list[Line2D | Artist] = []

        # self._peak_plotter = PeakPlotter(ax=ax)

    def draw_peak_mappings(
        self,
        signal: bool = False,
        maxima: bool = True,
        whh: bool = False,
        bases: bool = True,
    ) -> PeakMapPlotHandler:
        """
        draw all the information contained within the peak map overlaying the signal.
        """

        # default to None, initialise if selected by user. the output class contains
        # a method to prepare the overlay, skipping over attrs with value None to avoid
        # error

        plot_obj_signal = None
        plot_obj_maxima = None
        plot_obj_whh = None
        plot_obj_bases = None

        if signal:
            plot_obj_signal = self.draw_signal()
        if maxima:
            plot_obj_maxima = self.draw_maxima_hvplot()
        if whh:
            warnings.warn("whh plot not implemented")
            plot_obj_whh = None
        if bases:
            plot_obj_bases = self.draw_base_edges_hvplot()

        peak_map_plots = PeakMapPlotHandler(
            signal=plot_obj_signal,
            maxima=plot_obj_maxima,
            whh=plot_obj_whh,
            bases=plot_obj_bases,
        )

        peak_map_plots.overlay()

        return peak_map_plots

    def draw_base_edges_hvplot(self):
        """
        Draw open-ended triangles running from the left base to the maxima to the right base for each peak, grouped by p_idx. Requires some data manipulation such that each line will be defined by the three points mentioned, each labeled by the p_idx.

        """

        # a maxima value for each of the peak msnts
        contour_plotting_table = self.prepare_contour_plotting_table()

        # )  # fmt: skip
        base_edge_plot_obj = contour_plotting_table.plot(
            x="X_idx", y="X", by=["p_idx", "msnt"], label="base_edges"
        )

        return base_edge_plot_obj

    def prepare_contour_plotting_table(self):
        """
        Prepare a square (ish) table of each of the contour measurements 'msnt' with sides 'side' and maxima, with y (X) and x (X_idx) columns ready for plotting.

        The primary use is to draw lines reaching from the maxima of each peak down to the contour line sides in order to get a feel for how the algorithms are mapping the signal.
        """

        # assemble a maxima entry for each contour measurement of each peak. This will be appended to 'contour_line_bounds'.

        contour_bound_maximas = (
            self.peak_map.contour_line_bounds.pipe(pl.from_pandas)
            .select(pl.col([mp_defs.P_IDX, mp_defs.KEY_MSNT]))
            .unique()
            .join(
                how="left",
                other=self.peak_map.maxima.pipe(pl.from_pandas),
                on=mp_defs.P_IDX,
            )
            .select(
                [
                    mp_defs.P_IDX,
                    mp_defs.LOC,
                    mp_defs.KEY_MSNT,
                    mp_defs.DIM,
                    mp_defs.VALUE,
                ]
            )
        )

        # concat with contour_line_bounds

        concatenated_contour_maximas = pl.concat(
            [
                self.peak_map.contour_line_bounds.pipe(pl.from_pandas).with_columns(
                    pl.col(mp_defs.DIM).replace(
                        {mp_defs.KEY_X_IDX_ROUNDED: mp_defs.X_IDX}
                    )
                ),
                contour_bound_maximas,
            ]
        )

        concatenated_contour_maximas_pivot = (
            concatenated_contour_maximas.pivot(
                index=[mp_defs.P_IDX, mp_defs.LOC, mp_defs.KEY_MSNT],
                columns=mp_defs.DIM,
                values=mp_defs.VALUE,
            )
            .select(
                pl.col(
                    [
                        mp_defs.P_IDX,
                        mp_defs.KEY_MSNT,
                        mp_defs.LOC,
                        mp_defs.X_IDX,
                        mp_defs.X,
                    ]
                )
            )
            .sort([mp_defs.P_IDX, mp_defs.KEY_MSNT, mp_defs.LOC])
        )

        return concatenated_contour_maximas_pivot

    def draw_signal(self):
        """
        use hvplot namespace to produce a plot obj of the input X signal.
        """

        plot_obj = self._X.pipe(pl.from_pandas).plot(
            x="X_idx",
            y="X",
            label="X",
            title="X",
        )
        breakpoint()
        return plot_obj

    def draw_maxima_hvplot(self):
        plot_obj = (
            self.peak_map.maxima.pipe(pl.from_pandas)
            .pivot(columns="dim", index="p_idx", values="value")
            .plot(x="X_idx", y="X", label="maxima", kind="scatter", by="p_idx")
        )

        return plot_obj

    def __draw_maxima_matplotlib(self) -> Self:
        """
        call `PeakPlotter.plot_maxima` to draw the maximas on the internal Axes. Needs to first subset the peak_map to `maxima_x_y`.

        :return: _description_
        :rtype: Self
        """
        label = "maxima"
        color_key = "color"
        peak_map_pl = pl.from_pandas(self.peak_map)

        draw_annotated_maxima(
            peak_map=self.peak_map,
            ax=self.ax,
            x_key=self.maxima_idx_key,
            y_key=self.maxima_key,
            color_colname=self.color_key,
        )

        self.__add_proxy_artist_to_handles(
            label=label,
            line_2d=self.ax.lines[-1],
            color=get_first_value(peak_map_pl, color_key),
        )

        return self

    def __draw_base_edges(
        self,
    ) -> Self:
        """
        Draw lines connecting the left and right bases of each peak to demonstrate the space allocated to each peak.
        """
        label = "base maxima interpol."

        peak_map_pl = pl.from_pandas(self.peak_map)
        color_key = "color"

        draw_peak_base_edges(
            peak_map=self.peak_map,
            ax=self.ax,
            left_x_key="pb_left",
            right_x_key="pb_right",
            base_height_key="pb_height",
            maxima_x_key="X_idx",
            maxima_y_key="maxima",
            color_key="color",
        )
        self.__add_proxy_artist_to_handles(
            label=label,
            line_2d=self.ax.lines[-1],
            color=get_first_value(peak_map_pl, color_key),
        )

        return self

    def __draw_base_vertices(
        self,
    ) -> Self:
        color_key = "color"
        label = "bases"
        peak_map_pl: pl.DataFrame = pl.from_pandas(self.peak_map)

        draw_width_vertices(
            peak_map=self.peak_map,
            left_x_key="pb_left",
            right_x_key="pb_right",
            y_key="pb_height",
            marker="v",
            ax=self.ax,
            color_key=color_key,
            label=label,
        )

        self.__add_proxy_artist_to_handles(
            label=label,
            line_2d=self.ax.lines[-1],
            color=get_first_value(peak_map_pl, color_key),
        )

        return self

    def __show(self) -> Self:
        """
        display the current Axes.

        Graphics(?) are drawn onto the internal Axes object by other methods within this class. Calling this method uses up the current Axes.
        """

        add_handles_to_legend(ax=self.ax, handles=self.handles)
        plt.show()

        return self

    def __add_proxy_artist_to_handles(
        self,
        label: str,
        line_2d: Line2D,
        color,
    ) -> None:
        """
        Code lifted directly from matplotlib legend guide <https://matplotlib.org/stable/users/explain/axes/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists>
        """

        legend_proxy = set_legend_proxy_artist(label, line_2d, color)

        self.handles += [legend_proxy]


@pa.check_types
def assign_colors_to_p_idx(
    p_idx: pl.DataFrame,
) -> DataFrame[ColorMap]:
    """
    Create a table indexed by the peak_idx 'p_idx' containing a column 'color' which maps
    a distinct color to each peak. Use to generate a colorscheme and join to a table
    of plotting values as needed.
    """
    color_map = (
        p_idx.sort(by=mp_defs.P_IDX)
        .pipe(
            lambda df: df.with_columns(
                pl.Series(
                    name="color",
                    values=sns.color_palette(cc.glasbey_dark, n_colors=df.shape[0]),
                )
            )
        )
        .to_pandas()
        .pipe(ColorMap.validate, lazy=True)
        .pipe(DataFrame[ColorMap])
    )

    return color_map


def set_legend_proxy_artist(
    label: str,
    line_2d: Line2D,
    color,
) -> Line2D:
    """
    Creates an empty Line2D object to use as the marker for the legend entry.

    Used to generate a representation of a marker category when there are too
    many entries for a generic legend.

    :param label: the legend entry text
    :type label: str
    :param line_2d: a representative line_2d of the plotted data to extract marker
    information from. Ensures that the legend marker matches what was plotted.
    """

    # TODO: modify this to a more appropriate method of choosing a 'representative'
    marker = line_2d.get_marker()
    markersize = line_2d.get_markersize()
    markeredgewidth = line_2d.get_markeredgewidth()
    markeredgecolor = line_2d.get_markeredgecolor()
    ls = line_2d.get_linestyle()

    proxy = Line2D(
        [],
        [],
        marker=marker,
        markersize=markersize,
        color=color,
        markeredgewidth=markeredgewidth,
        markeredgecolor=markeredgecolor,
        label=label,
        ls=ls,
    )
    return proxy


def get_legend_handle_labels(ax):
    return [handle.get_label() for handle in ax.get_legend_handles_labels()[0]]


@pa.check_types
def draw_annotated_maxima(
    peak_map: DataFrame[PeakMapWideColored],
    ax: Axes,
    x_key: str,
    y_key: str,
    color_colname: str,
) -> None:
    peak_map_pl: pl.DataFrame = pl.from_pandas(peak_map)

    row: dict

    for row in peak_map_pl.iter_rows(named=True):
        ax.plot(
            row[x_key],
            row[y_key],
            marker="o",
            c=row[color_colname],
            markeredgecolor="black",
            label="_",
            ls="",
        )

        ax.annotate(
            text=row["p_idx"],
            xy=(row[x_key], row[y_key]),
            ha="center",
            va="top",
            textcoords="offset pixels",
            xytext=(0, 40),
        )


@pa.check_types
def draw_width_vertices(
    peak_map: DataFrame[PeakMapWideColored],
    left_x_key: str,
    right_x_key: str,
    y_key: str,
    marker: str,
    ax: Axes,
    color_key: str,
    label: str = "width",
    plot_kwargs: dict = {},
) -> None:
    """
    Main interface for plotting the width ips as points. Can be used to plot the intersect
    between the countour line and signal for any rel height input in `scipy.signal.peak_widths`
    """

    peak_map_pl = pl.from_pandas(peak_map)

    for row in peak_map_pl.iter_rows(named=True):
        ax.plot(
            (row[left_x_key], row[right_x_key]),
            (row[y_key], row[y_key]),
            c=row[color_key],
            marker=marker,
            markeredgecolor="black",
            ls="",
            label="_",
            **plot_kwargs,
        )


@typechecked
def draw_line_between_two_points(
    ax: Axes,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    ls: str,
    color: tuple[float, float, float],
):
    """
    Given two points defined by `x1`, `y1`, `x2`, `y2`, draw a line betwen them with style `ls`
    on `ax`.
    """
    x = (x1, x2)
    y = (y1, y2)

    ax.plot(
        x,
        y,
        color=color,
        ls=ls,
    )


@pa.check_types
def draw_peak_base_edges(
    peak_map: DataFrame[PeakMapWideColored],
    ax,
    left_x_key: str,
    right_x_key: str,
    base_height_key: str,
    maxima_x_key: str,
    maxima_y_key: str,
    color_key: str,
):
    """
    For each peak in `peak_map`, draw lines connecting the maxima to the specified
    base on either side. This is used as a tool to observe peak overlap as per how
    the `hplc_py` defines a peak base as the width at rel_height = 0.associated

    Uses `pipe_join_width_maxima_long` to arrange the peak_map as needed, one row
    per side of a peak (i.e. pairs of the same peak, 'p_idx') the maxima x and y of
    that peak in each row. Check the schema Width_Maxima_Join for more info.
    """
    peak_map_pl = pl.from_pandas(peak_map)

    for row in peak_map_pl.iter_rows(named=True):
        color = tuple(row[color_key])

        draw_line_between_two_points(
            ax=ax,
            x1=row[left_x_key],
            y1=row[base_height_key],
            x2=row[maxima_x_key],
            y2=row[maxima_y_key],
            color=color,
            ls="--",
        )
        draw_line_between_two_points(
            ax=ax,
            x1=row[maxima_x_key],
            y1=row[maxima_y_key],
            x2=row[right_x_key],
            y2=row[base_height_key],
            color=color,
            ls="--",
        )


def add_handles_to_legend(
    ax: Axes,
    handles: list[Line2D | Artist],
) -> None:
    """
    adds custom handles to ax legend. Call prior to plotting.
    """

    curr_handles, labels = ax.get_legend_handles_labels()

    new_handles: list[Line2D | Artist] = curr_handles + handles

    ax.legend(handles=new_handles)


def get_first_value(
    df: pl.DataFrame,
    key: str,
):
    """
    Given a dataframe with column `key` extract the first value in `key` and return
    """
    first_val = df.select(pl.col(key).first()).item()
    return first_val

    breakpoint()
