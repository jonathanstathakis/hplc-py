import polars as pl
import pytest
from hplc_py.map_windows.viz import UI_WindowMapViz
from hplc_py.map_windows.viz import draw_peak_windows
from hplc_py.hplc_py_typing.hplc_py_typing import X_Schema, PeakMapWide
from hplc_py.map_windows.typing import X_Windowed
from pandera.typing import DataFrame
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def test_window_map_viz(
    X: DataFrame[X_Schema],
    peak_map: DataFrame[PeakMapWide],
    X_w: DataFrame[X_Windowed],
):
    window_viz = UI_WindowMapViz(
        peak_map=peak_map,
        X_w=X_w,
        ax=plt.gca(),
        )
    window_viz.draw_signal().draw_maxima().draw_peak_windows().draw_base_edges().show()
