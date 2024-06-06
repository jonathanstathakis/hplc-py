"""
Pure functions for plotting information from `peak_map`.
"""

import pandas as pd
import deprecation
import holoviews as hv
import numpy as np
import hvplot
import seaborn as sns
import colorcet as cc
import pandera as pa
import polars as pl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from pandera.typing import DataFrame
from . import contour_line_bounds
from . import definitions as mp_defs
from hplc_py.hplc_py_typing.hplc_py_typing import ColorMap
from .schemas import PeakMapWideColored

from dataclasses import dataclass, fields
from typing import Any

import logging
import panel as pn


def exception_handler(ex):

    logging.error("INFO", exc_info=ex)

    pn.state.notifications.error("RAR: %s" % ex)


pn.extension(exception_handler=exception_handler, notifications=True)

class PeakMapViz:
    
    def __init__(self, peak_map):
        self.peak_map = peak_map
        
    
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


def assign_colors_to_p_idx(
    p_idx: pd.DataFrame,
) -> DataFrame[ColorMap]:
    """
    Create a table indexed by the peak_idx 'p_idx' containing a column 'color' which maps
    a distinct color to each peak. Use to generate a colorscheme and join to a table
    of plotting values as needed.
    """

    color_map = (
        p_idx.pipe(pl.from_pandas)
        .sort(by=mp_defs.P_IDX)
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


@deprecation.deprecated
def __set_legend_proxy_artist(
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


@deprecation.deprecated
def __get_legend_handle_labels(ax):
    return [handle.get_label() for handle in ax.get_legend_handles_labels()[0]]


@deprecation.deprecated
def __draw_annotated_maxima(
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


@deprecation.deprecated
@pa.check_types
def __draw_width_vertices(
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


@deprecation.deprecated
def __draw_line_between_two_points(
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


@deprecation.deprecated
@pa.check_types
def __draw_peak_base_edges(
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

        __draw_line_between_two_points(
            ax=ax,
            x1=row[left_x_key],
            y1=row[base_height_key],
            x2=row[maxima_x_key],
            y2=row[maxima_y_key],
            color=color,
            ls="--",
        )
        __draw_line_between_two_points(
            ax=ax,
            x1=row[maxima_x_key],
            y1=row[maxima_y_key],
            x2=row[right_x_key],
            y2=row[base_height_key],
            color=color,
            ls="--",
        )


@deprecation.deprecated
def __add_handles_to_legend(
    ax: Axes,
    handles: list[Line2D | Artist],
) -> None:
    """
    adds custom handles to ax legend. Call prior to plotting.
    """

    curr_handles, labels = ax.get_legend_handles_labels()

    new_handles: list[Line2D | Artist] = curr_handles + handles

    ax.legend(handles=new_handles)


@deprecation.deprecated
def __get_first_value(
    df: pl.DataFrame,
    key: str,
):
    """
    Given a dataframe with column `key` extract the first value in `key` and return
    """
    first_val = df.select(pl.col(key).first()).item()
    return first_val

    breakpoint()
