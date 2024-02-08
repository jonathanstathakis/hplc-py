"""
Test whether the PeakMapViz class correctly plots the peak map values onto the signal.

Works by passing a pre-generated Axes object between the fixtures, iteratively drawing
onto the Axes to produce the expected overlay plots. `test_chain_plot` produces
the same thing but via method chaining from the `loaded_pmv` object.

Note: Uses time rather than t_idx as the x-axis units.
"""
import pytest
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandera.typing import Series, DataFrame
from numpy import float64
import matplotlib.pyplot as plt
from hplc_py.show import SignalPlotter, PeakMapViz
from hplc_py.hplc_py_typing.hplc_py_typing import PeakMap, X_Schema
from hplc_py.map_signals.map_peaks.map_peaks_viz import PeakMapInterface, PeakPlotter, WidthPlotter

def test_signal_plot(
    plot_signal,
) -> None:
    plt.show()


def test_plot_peaks(
    peak_map_fig: Figure,
    plot_signal,
    plot_peaks,
):
    plt.show()


def test_plot_whh(
    plot_signal,
    plot_peaks,
    plot_whh,
) -> None:
    plt.show()
    
@pytest.fixture
def plot_whh(
    peak_map_ax,
    loaded_pmv: PeakMapViz,
):
    loaded_pmv.plot_whh(
        y_colname=str(PeakMap.whh_height),
        left_colname=str(PeakMap.whh_left),
        right_colname=str(PeakMap.whh_right),
        ax=peak_map_ax,
    )



def test_plot_pb(
    plot_signal,
    plot_peaks,
    plot_whh,
    plot_pb,
) -> None:
    plt.show()


def test_chain_plot(
    loaded_pmv: PeakMapViz,
    plot_signal,
) -> None:
    """
    Test whether chaining the plot methods produces the expected drawing

    :param loaded_pmv: PeakMapViz object loaded with data
    :type loaded_pmv: PeakMapViz
    :param plot_signal: A fixture plotting the signal on the axis loaded into `loaded_pmv`
    :type plot_signal: None
    """
    (
        loaded_pmv.plot_whh(
            str(PeakMap.whh_height), str(PeakMap.whh_left), str(PeakMap.whh_right)
        ).plot_bases(
            str(PeakMap.pb_height),
            str(PeakMap.pb_left),
            str(PeakMap.pb_right))
    )

    plt.show()

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

@pytest.fixture
def plot_signal(
    X: DataFrame[X_Schema],
    x_colname: str,
    y_colname: str,
    peak_map_ax: Axes,
):
    X = X.reset_index(names='X_idx')
    
    SignalPlotter().plot_signal(
        ax=peak_map_ax,
        df=X,
        x_colname=x_colname,
        y_colname=y_colname,
        label=y_colname,
    )

    return None


@pytest.fixture
def loaded_pmv(
    peak_map: DataFrame[PeakMap],
    peak_map_ax: Axes,
    x_colname: str,
) -> PeakMapViz:
    pmv = PeakMapViz(df=peak_map, x_colname=x_colname, ax=peak_map_ax)

    return pmv

@pytest.fixture
def plot_peaks(
    loaded_pmv: PeakMapViz,
    maxima_colname: str,
) -> None:
    loaded_pmv.plot_peaks(pmax_col=maxima_colname)
    
@pytest.fixture
def plot_pb(
    peak_map_ax,
    loaded_pmv: PeakMapViz,
):
    loaded_pmv.plot_bases(
        y_colname=str(PeakMap.pb_height),
        left_colname=str(PeakMap.pb_left),
        right_colname=str(PeakMap.pb_right),
        ax=peak_map_ax,
    )
    
@pytest.fixture
def peakplotter(
    peak_map_ax: Axes,
    peak_map: DataFrame[PeakMap],
    )->PeakPlotter:
    peakplotter = PeakPlotter(ax=peak_map_ax, df=peak_map)
    return peakplotter

def test_peak_plotter(
    peakplotter: PeakPlotter,
    x_colname: str,
    y_colname: str,
    ):
    peakplotter.plot_peaks(
        x_colname=x_colname,
        y_colname=y_colname,
    )
    plt.show()

@pytest.fixture
def width_plotter(
    peak_map_ax: Axes,
    peak_map: DataFrame[PeakMap],
)->WidthPlotter:
    
    widthplotter = WidthPlotter(ax=peak_map_ax, df=peak_map)
    
    return widthplotter
    
def test_width_plotter(
    plot_signal,
    width_plotter: WidthPlotter
)->None:
    width_plotter.plot_widths(
        y_colname=str(PeakMap.pb_height),
        left_x_colname=str(PeakMap.pb_left),
        right_x_colname=str(PeakMap.pb_right),
        marker="v"
    )
    plt.show()
    
    

def test_peak_map(
    peak_map
):
    breakpoint()
    pass


@pytest.fixture
def pmi(peak_map: DataFrame[PeakMap], X: DataFrame[X_Schema])->PeakMapInterface:
    pmi = PeakMapInterface(peak_map=peak_map, X=X)
    return pmi


class TestPeakMapInterface:
    def test_Pipe_Peak_Widths_To_Long(self,
                                      peak_map: DataFrame[PeakMap],
                                      ):
        from hplc_py.map_signals.map_peaks.map_peaks_viz import Pipe_Peak_Widths_To_Long
        
        pipe_peak_widths_to_long = Pipe_Peak_Widths_To_Long()
        
        pipe_peak_widths_to_long.load_pipeline(
            peak_map=peak_map
        )
        pipe_peak_widths_to_long.run_pipeline()
        breakpoint()
        
    def test_pipe_peak_maxima_to_long(
        self,
        peak_map: DataFrame[PeakMap]
    )->None:
        from hplc_py.map_signals.map_peaks.map_peaks_viz import Pipe_Peak_Maxima_To_Long
        
        pipe_peak_maxima_to_long = Pipe_Peak_Maxima_To_Long()
        
        (pipe_peak_maxima_to_long
         .load_pipeline(peak_map)
         .run_pipeline()
         )
        