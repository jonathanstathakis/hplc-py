"""
Test whether the PeakMapWideViz class correctly plots the peak map values onto the signal.

Works by passing a pre-generated Axes object between the fixtures, iteratively drawing
onto the Axes to produce the expected overlay plots. `test_chain_plot` produces
the same thing but via method chaining from the `loaded_pmv` object.

Note: Uses time rather than t_idx as the x-axis units.
"""

import distinctipy
from hplc_py.map_signals.map_peaks.peak_map_plot_ui import UI_PlotPeakMapWide
from hplc_py.map_signals.map_peaks.map_peaks_viz_pipelines import (
    Pipe_Peak_Maxima_To_Long,
)
import polars as pl
import pandera as pa
import pytest
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandera.typing import Series, DataFrame
from numpy import float64
import matplotlib.pyplot as plt
import matplotlib as mpl
from hplc_py.hplc_py_typing.hplc_py_typing import PeakMapWide, X_Schema
from hplc_py.map_signals.map_peaks.map_peaks_viz_schemas import (
    PM_Width_In_X,
    PM_Width_In_Y,
    PM_Width_Long_Out_X,
    PM_Width_Long_Out_Y,
    PM_Width_Long_Joined,
    Maxima_X_Y,
    Width_Maxima_Join,
    ColorMap,
)

from hplc_py.map_signals.map_peaks.map_peaks_viz_pipelines import (
    Pipe_Peak_Widths_To_Long,
    Pipeline_Join_Width_Maxima_Long,
    Pipeline_Peak_Map_Interface,
)


@pytest.fixture
def peak_map_long(peak_map: DataFrame[PeakMapWide]) -> pl.DataFrame:

    peak_map_long = pl.DataFrame()
    peak_map_pl = pl.from_pandas(peak_map)
    # peak_map_long = peak_map.melt(id_va)
    breakpoint()
    return peak_map_long


def test_peak_map_long(peak_map_long):
    breakpoint()


# def test_plot_whh(
#     plot_signal,
#     plot_peaks,
#     plot_whh,
# ) -> None:
#     plt.show()


# @pytest.fixture
# def plot_whh(
#     peak_map_ax,
#     loaded_pmv: PeakMapWideViz,
# ):
#     loaded_pmv.plot_whh(
#         y_colname=str(PeakMapWide.whh_height),
#         left_colname=str(PeakMapWide.whh_left),
#         right_colname=str(PeakMapWide.whh_right),
#         ax=peak_map_ax,
#     )


# def test_plot_pb(
#     plot_signal,
#     plot_peaks,
#     plot_whh,
#     plot_pb,
# ) -> None:
#     plt.show()


# def test_chain_plot(
#     loaded_pmv: PeakMapWideViz,
#     plot_signal,
# ) -> None:
#     """
#     Test whether chaining the plot methods produces the expected drawing

#     :param loaded_pmv: PeakMapWideViz object loaded with data
#     :type loaded_pmv: PeakMapWideViz
#     :param plot_signal: A fixture plotting the signal on the axis loaded into `loaded_pmv`
#     :type plot_signal: None
#     """
#     (
#         loaded_pmv.plot_whh(
#             str(PeakMapWide.whh_height), str(PeakMapWide.whh_left), str(PeakMapWide.whh_right)
#         ).plot_bases(
#             str(PeakMapWide.pb_height), str(PeakMapWide.pb_left), str(PeakMapWide.pb_right)
#         )
#     )

#     plt.show()


@pytest.fixture
def peak_map_fig():
    return plt.figure()


@pytest.fixture
def peak_map_ax(peak_map_fig):
    return peak_map_fig.subplots()


@pytest.fixture
def x_colname():
    return "X_idx"


@pytest.fixture
def y_colname():
    return "X"


@pytest.fixture
def maxima_colname():
    return "maxima"


def test_plot_signal_Signal_Plotter(
    X: DataFrame[X_Schema],
    x_colname: str,
    y_colname: str,
    peak_map_ax: Axes,
):
    X = X.reset_index(names="X_idx")

    SignalPlotter(df=X, ax=peak_map_ax).plot_signal(
        x_colname=x_colname,
        y_colname=y_colname,
        label=y_colname,
    )

    plt.show()

    return None


@pytest.fixture
def widths_long_xy(
    peak_map: DataFrame[PeakMapWide],
) -> DataFrame[PM_Width_Long_Joined]:

    pipe_peak_widths_to_long = Pipe_Peak_Widths_To_Long()

    pipe_peak_widths_to_long.load_pipeline(peak_map=peak_map)
    pipe_peak_widths_to_long.run_pipeline()

    return pipe_peak_widths_to_long.widths_long_xy


@pytest.fixture
def maxima_x_y(
    peak_map: DataFrame[PeakMapWide],
) -> DataFrame[Maxima_X_Y]:

    pipe_peak_maxima_to_long = Pipe_Peak_Maxima_To_Long()

    (pipe_peak_maxima_to_long.load_pipeline(peak_map).run_pipeline())
    return pipe_peak_maxima_to_long.maxima_x_y


@pytest.fixture
def colors(peak_map: DataFrame[PeakMapWide]):
    colors = distinctipy.get_colors(len(peak_map))

    return colors


def test_draw_maxima(
    colors: list[tuple[float, float, float]],
    maxima_x_y: DataFrame[Maxima_X_Y],
    peak_map_ax: Axes,
) -> None:

    peakplotter = MaximaPlotter(ax=peak_map_ax, peak_map=maxima_x_y, colors=colors)

    peakplotter.draw_maxima()


@pytest.fixture
def width_maxima_join(
    widths_long_xy: DataFrame[PM_Width_Long_Joined],
    maxima_x_y: DataFrame[Maxima_X_Y],
) -> DataFrame[Width_Maxima_Join]:
    pipeline_join_width_maxima_long = Pipeline_Join_Width_Maxima_Long()
    (
        pipeline_join_width_maxima_long.load_pipeline(
            widths_long_xy=widths_long_xy.to_pandas(),
            maxima_x_y=maxima_x_y.to_pandas(),
        ).run_pipeline()
    )

    return pipeline_join_width_maxima_long.width_maxima_join


# @pytest.fixture
# def loaded_pmv(
#     peak_map: DataFrame[PeakMapWide],
#     peak_map_ax: Axes,
#     x_colname: str,
# ) -> PeakMapWideViz:
#     pmv = PeakMapWideViz(df=peak_map, x_colname=x_colname, ax=peak_map_ax)

#     return pmv


# @pytest.fixture
# def plot_pb(
#     peak_map_ax,
#     loaded_pmv: PeakMapWideViz,
# ):
#     loaded_pmv.plot_bases(
#         y_colname=str(PeakMapWide.pb_height),
#         left_colname=str(PeakMapWide.pb_left),
#         right_colname=str(PeakMapWide.pb_right),
#         ax=peak_map_ax,
#     )


@pytest.fixture
def peakplotter(
    peak_map_ax: Axes,
    peak_map: DataFrame[PeakMapWide],
) -> MaximaPlotter:
    peakplotter = MaximaPlotter(ax=peak_map_ax, peak_map=peak_map)
    return peakplotter


def test_peak_plotter(
    peakplotter: MaximaPlotter,
    x_colname: str,
    y_colname: str,
):
    peakplotter.draw_maxima(
        x_colname=x_colname,
        y_colname=y_colname,
    )
    plt.show()


@pytest.fixture
def width_plotter(
    peak_map_ax: Axes,
    peak_map: DataFrame[PeakMapWide],
) -> WidthPlotter:

    widthplotter = WidthPlotter(ax=peak_map_ax, peak_map=peak_map)

    return widthplotter


def test_width_plotter(plot_signal_Signal_Plotter, width_plotter: WidthPlotter) -> None:
    width_plotter.plot_widths_vertices(
        y_colname=str(PeakMapWide.pb_height),
        left_x_colname=str(PeakMapWide.pb_left),
        right_x_colname=str(PeakMapWide.pb_right),
        marker="v",
    )
    plt.show()


@pytest.fixture
def pmi(
    peak_map: DataFrame[PeakMapWide], X: DataFrame[X_Schema]
) -> Pipeline_Peak_Map_Interface:
    pmi = Pipeline_Peak_Map_Interface(peak_map=peak_map, X=X)
    return pmi


class TestPipeline_Peak_Map_Interface:

    def test_Pipe_Peak_Widths_To_Long(
        self, widths_long_xy: DataFrame[PM_Width_Long_Joined]
    ):
        PM_Width_Long_Joined(widths_long_xy.to_pandas(), lazy=True)
        pass

    def test_pipe_peak_maxima_to_long(
        self,
        maxima_x_y: DataFrame[Maxima_X_Y],
    ) -> None:
        Maxima_X_Y(maxima_x_y.to_pandas(), lazy=True)

    def test_pipe_join_width_maxima_long(
        self,
        width_maxima_join,
    ):
        Width_Maxima_Join.validate(width_maxima_join.to_pandas(), lazy=True)
        pass

    @pytest.fixture
    @pa.check_types
    def peak_map_plot_data(
        self, peak_map: DataFrame[PeakMapWide]
    ) -> DataFrame[Width_Maxima_Join]:

        pipeline_peak_map_interface = Pipeline_Peak_Map_Interface()
        peak_map_plot_data = (
            pipeline_peak_map_interface.load_pipeline(peak_map=peak_map)
            .run_pipeline()
            .peak_map_plot_data
        )
        return peak_map_plot_data

    def test_pipeline_peak_map_interface(
        self, peak_map_plot_data: DataFrame[Width_Maxima_Join]
    ):
        Width_Maxima_Join.validate(peak_map_plot_data.to_pandas(), lazy=True)


def test_assign_colors_to_peakmap(peak_map: DataFrame[PeakMapWide]) -> None:
    """
    Assign a color to each peak_idx, returning a new frame 'peak_map_colored'
    """

    from hplc_py.map_signals.map_peaks.peakplotfuncs import assign_colors_to_p_idx

    colormap = assign_colors_to_p_idx(peak_map)

    ColorMap.validate(colormap, lazy=True)
