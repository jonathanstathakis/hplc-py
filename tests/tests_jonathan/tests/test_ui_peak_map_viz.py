import pandera as pa
from hplc_py.map_signals.map_peaks.peak_map_plot_ui import UI_PlotPeakMapWide
import polars as pl
import pytest
from pandera.typing import DataFrame
from hplc_py.hplc_py_typing.hplc_py_typing import (
    PeakMapWide,
    X_Schema,
    PeakMapWide,
    PeakMapWideColored,
)
from hplc_py.map_signals.map_peaks.map_peaks_viz_schemas import ColorMap
from hplc_py.map_signals.map_peaks.peakplotfuncs import (
    assign_colors_to_p_idx,
    pivot_peak_map,
    join_peak_map_colors,
)


@pytest.fixture
def color_map(peak_map: DataFrame[PeakMapWide]) -> DataFrame[ColorMap]:

    colormap = assign_colors_to_p_idx(peak_map=peak_map)

    return colormap


@pytest.fixture
def peak_map_wide(peak_map: DataFrame[PeakMapWide]) -> DataFrame[PeakMapWide]:
    # peak_map_wide = pivot_peak_map(peak_map=peak_map)
    # return peak_map_wide
    return peak_map


@pytest.fixture
@pa.check_types
def peak_map_wide_colored(
    peak_map_wide: DataFrame[PeakMapWide], color_map: DataFrame[ColorMap]
) -> DataFrame[PeakMapWideColored]:

    peak_map_wide_colored = join_peak_map_colors(
        peak_map_wide=peak_map_wide, color_map=color_map
    )

    return peak_map_wide_colored


@pa.check_types
def test_join_colors_peak_map(
    peak_map_wide_colored: DataFrame[PeakMapWideColored],
) -> None:

    PeakMapWideColored.validate(peak_map_wide_colored)


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
        (ui_plot_peak_map
        .draw_signal()
        .draw_maxima()
        .draw_base_vertices()
        .show()
        )

    def test_draw_edges(
        self,
        ui_plot_peak_map: UI_PlotPeakMapWide,
    ):
        ui_plot_peak_map.draw_signal().draw_maxima()
        ui_plot_peak_map.draw_base_edges()
        ui_plot_peak_map.show()
