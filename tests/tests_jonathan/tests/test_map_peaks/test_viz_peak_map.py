import hvplot
import pandera as pa
from hplc_py.map_peaks.map_peaks import MapPeaks
import polars as pl
import pytest
from pandera.typing import DataFrame
from hplc_py.common.common_schemas import X_Schema
from hplc_py.hplc_py_typing.hplc_py_typing import (
    ColorMap,
)
from hplc_py.map_peaks.schemas import PeakMap, PeakMapWideColored

from hplc_py.map_peaks.viz import (
    __assign_colors_to_p_idx,
    VizPeakMapFactory,
)


@pytest.fixture
def color_map(peak_map: DataFrame[PeakMap]) -> DataFrame[ColorMap]:

    colormap = __assign_colors_to_p_idx(peak_map=peak_map)

    return colormap


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
        self, X_data, peak_map: DataFrame[PeakMap]
    ) -> VizPeakMapFactory:
        plot_peak_map = VizPeakMapFactory(
            X=pl.from_pandas(X_data).with_row_index("X_idx"), peak_map=peak_map
        )
        return plot_peak_map

    def test_draw_maxima(
        self,
        ui_plot_peak_map: VizPeakMapFactory,
    ):
        """
        test drawing the maximas through the PlotPeakMapWide UI
        """

        ui_plot_peak_map.draw_signal().draw_maxima().show()

    def test_draw_widths_vertices(
        self,
        ui_plot_peak_map: VizPeakMapFactory,
    ):
        (ui_plot_peak_map.draw_signal().draw_maxima().draw_base_vertices().show())

    def test_draw_edges(
        self,
        ui_plot_peak_map: VizPeakMapFactory,
    ):
        ui_plot_peak_map.draw_signal().draw_maxima()
        ui_plot_peak_map.__draw_base_edges()
        ui_plot_peak_map.__show()


def test_viz_peak_map_hvplot(X_bcorr, map_peaks_mapped: MapPeaks):

    plot_obj = map_peaks_mapped.plot.draw_signal()
    hvplot.show(plot_obj)


def test_viz_draw_peak_map(X_bcorr, map_peaks_mapped: MapPeaks):

    map_peaks_mapped.plot.draw_peak_mappings()


def test_viz_prominence_data(
    X_bcorr: DataFrame[X_Schema],
):

    mp = PeakMapper()
    mp.fit(X=X_bcorr)
    mp.transform()
    mp.viz_peak_prom_data()
    pass
