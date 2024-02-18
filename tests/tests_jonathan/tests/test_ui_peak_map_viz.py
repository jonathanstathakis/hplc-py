import pandera as pa

import polars as pl
import pytest
from pandera.typing import DataFrame
from hplc_py.common_schemas import X_Schema
from hplc_py.hplc_py_typing.hplc_py_typing import (
    PeakMapWide,
    PeakMapWideColored,
    ColorMap,
)

from hplc_py.map_peaks.viz import (
    assign_colors_to_p_idx,
    join_peak_map_colors,
    UI_PlotPeakMapWide,
)


@pytest.fixture
def color_map(peak_map: DataFrame[PeakMapWide]) -> DataFrame[ColorMap]:

    colormap = assign_colors_to_p_idx(peak_map=peak_map)

    return colormap


@pytest.fixture
@pa.check_types
def peak_map_colored(
    peak_map: DataFrame[PeakMapWide], color_map: DataFrame[ColorMap]
) -> DataFrame[PeakMapWideColored]:

    peak_map_colored = join_peak_map_colors(peak_map=peak_map, color_map=color_map)

    return peak_map_colored


@pa.check_types
def test_join_colors_peak_map(
    peak_map_colored: DataFrame[PeakMapWideColored],
) -> None:

    PeakMapWideColored.validate(peak_map_colored)


class Test_UI_Plot_Peak_Map:
    """
    test the higher level viz via PlotPeakMapWide class.
    """

    @pytest.fixture
    def ui_plot_peak_map(
        self, X: DataFrame[X_Schema], peak_map: DataFrame[PeakMapWide]
    ) -> UI_PlotPeakMapWide:
        plot_peak_map = UI_PlotPeakMapWide(
            X=pl.from_pandas(X).with_row_index("X_idx"), peak_map=peak_map
        )
        return plot_peak_map

    def test_draw_maxima(
        self,
        ui_plot_peak_map: UI_PlotPeakMapWide,
    ):
        """
        test drawing the maximas through the PlotPeakMapWide UI
        """

        ui_plot_peak_map.draw_signal().draw_maxima().show()

    def test_draw_widths_vertices(
        self,
        ui_plot_peak_map: UI_PlotPeakMapWide,
    ):
        (ui_plot_peak_map.draw_signal().draw_maxima().draw_base_vertices().show())

    def test_draw_edges(
        self,
        ui_plot_peak_map: UI_PlotPeakMapWide,
    ):
        ui_plot_peak_map.draw_signal().draw_maxima()
        ui_plot_peak_map.draw_base_edges()
        ui_plot_peak_map.show()
