from hplc_py.common.common_schemas import X_Schema
from hplc_py.map_windows.viz import WindowMapViz
from hplc_py.map_peaks.schemas import PeakMap
from hplc_py.map_windows.schemas import X_Windowed
from pandera.typing import DataFrame
import matplotlib.pyplot as plt


def test_window_map_viz(
    X_data,
    peak_map: DataFrame[PeakMap],
    X_windowed,
):
    window_viz = WindowMapViz(
        peak_map=peak_map,
        X_w=X_windowed,
        ax=plt.gca(),
    )
    window_viz.draw_signal().draw_maxima().draw_peak_windows().draw_base_edges().show()
