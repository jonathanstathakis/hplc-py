from matplotlib.lines import Line2D
import polars as pl
from typing import Self
from pandera.typing import DataFrame
from hplc_py.hplc_py_typing.hplc_py_typing import PeakMapWide, X_Schema

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist


class UI_PlotPeakMapWide:
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
        peak_map: DataFrame[PeakMapWide],
        ax: Axes = plt.gca(),
    ):
        """
        Provide an Axes object to draw on. if not provided, will draw on the current
        Axes
        """
        self.ax = ax
        self.X = X
        self.peak_map = join_peak_map_colors(peak_map, assign_colors_to_p_idx(peak_map))

        self.maxima_idx_key = "X_idx"
        self.maxima_key = "maxima"
        self.color_key = "color"

        self.handles: list[Line2D | Artist] = []

        # self._peak_plotter = PeakPlotter(ax=ax)

    def draw_signal(self) -> Self:
        """
        Call `SignalPlotter.plot_signal` to draw the input signal onto the internal Axes object.
        """
        plot_signal(
            self.X,
            x_colname="X_idx",
            y_colname="X",
            label="X",
            ax=self.ax,
        )
        return self

    def draw_maxima(self) -> Self:
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

        self.add_proxy_artist_to_handles(
            label=label,
            line_2d=self.ax.lines[-1],
            color=get_first_value(peak_map_pl, color_key),
        )

        return self

    def draw_base_edges(
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
        self.add_proxy_artist_to_handles(
            label=label,
            line_2d=self.ax.lines[-1],
            color=get_first_value(peak_map_pl, color_key),
        )

        return self

    def draw_base_vertices(
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

        self.add_proxy_artist_to_handles(
            label=label,
            line_2d=self.ax.lines[-1],
            color=get_first_value(peak_map_pl, color_key),
        )

        return self

    def show(self) -> Self:
        """
        display the current Axes.

        Graphics(?) are drawn onto the internal Axes object by other methods within this class. Calling this method uses up the current Axes.
        """

        add_handles_to_legend(ax=self.ax, handles=self.handles)
        plt.suptitle("testplot")
        plt.show()

        return self

    def add_proxy_artist_to_handles(
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


"""
Pure functions for plotting information from `peak_map`.
"""

import seaborn as sns
import colorcet as cc
import pandas as pd
import pandera as pa
import polars as pl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from pandera.typing import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import PeakMapWide
from typeguard import typechecked

from hplc_py.hplc_py_typing.hplc_py_typing import PeakMapWideColored, ColorMap


def plot_signal(
    df: pd.DataFrame | pl.DataFrame,
    x_colname: str,
    y_colname: str,
    label: str,
    ax: Axes,
    line_kwargs: dict = {},
) -> None:

    sig_x = df[x_colname]
    sig_y = df[y_colname]

    line = ax.plot(sig_x, sig_y, label=label, **line_kwargs)
    ax.legend(handles=line)


@pa.check_types
def assign_colors_to_p_idx(
    peak_map: DataFrame[PeakMapWide],
) -> DataFrame[ColorMap]:
    """
    Create a table indexed by the peak_idx 'p_idx' containing a column 'color' which maps
    a distinct color to each peak. Use to generate a colorscheme and join to a table
    of plotting values as needed.
    """

    peak_map_pl: pl.DataFrame = pl.from_pandas(peak_map)

    p_idx: pl.DataFrame = peak_map_pl.select("p_idx").unique(maintain_order=True)

    colors = sns.color_palette(cc.glasbey_dark, n_colors=41)

    # if you dont declare it as a Series the constructor broadcasts the list of tuples
    # to every row.

    color_table: pl.DataFrame = p_idx.with_columns(
        color=pl.Series(name="color", values=colors)
    )

    return color_table.to_pandas()


@pa.check_types
def pivot_peak_map(peak_map: DataFrame[PeakMapWide]) -> DataFrame[PeakMapWide]:

    peak_map_wide = peak_map.pivot(index="p_idx", columns="prop", values="value")

    return peak_map_wide


@pa.check_types
def join_peak_map_colors(
    peak_map_wide: DataFrame[PeakMapWide],
    color_map: DataFrame[ColorMap],
) -> DataFrame[PeakMapWideColored]:
    """
    Joins a wide peak map with unique peak indexes with the pre-determined color mapping
    """

    peak_map_wide_pl: pl.DataFrame = pl.from_pandas(peak_map_wide)
    color_map_pl: pl.DataFrame = pl.from_pandas(color_map)

    peak_map_colored_pl = peak_map_wide_pl.join(
        color_map_pl, on="p_idx", how="left", validate="1:1"
    )

    peak_map_colored = peak_map_colored_pl.to_pandas()

    return peak_map_colored


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
