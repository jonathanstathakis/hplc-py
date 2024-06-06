from typing import Self
from hplc_py.common.common_schemas import X_Schema
from hplc_py.map_signal.map_peaks import definitions as mp_defs
from hplc_py.map_signal.map_peaks.peak_map_output import PeakMap
from hplc_py.map_signal.map_peaks.viz_matplotlib import (
    PeakMapViz,
    __add_handles_to_legend,
    __draw_annotated_maxima,
    __draw_peak_base_edges,
    __draw_width_vertices,
    __get_first_value,
    __set_legend_proxy_artist,
    assign_colors_to_p_idx,
)


import deprecation
import matplotlib.pyplot as plt
import pandas as pd
import pandera as pa
import polars as pl
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from pandera.typing import DataFrame


import warnings


class VizPeakMapFactory:
    """
    Provides methods for plotting the maxima as a scatter, the WHH as scatter, and the bases as lines drawn from the maxima of the peak down to each base. Each method will provide a hvplot `holoviews.core.overlay.NDOverlay` object which will allow for downstream plot composition.
    """

    @pa.check_types
    def _draw_peak_mappings(
        self,
        X: DataFrame[X_Schema],
        peak_map: PeakMap,
        signal: bool = True,
        maxima: bool = True,
        whh: bool = False,
        base: str = "pb",
    ) -> PeakMapViz:
        """
        draw all the information contained within the peak map overlaying the signal.

        :base: choose from "pb", "whh", "prom", "all", or "none"
        """

        if not peak_map:
            breakpoint()
            raise TypeError("dont expect None for 'peak_map")
        # default to None, initialise if selected by user. the output class contains
        # a method to prepare the overlay, skipping over attrs with value None to avoid
        # error
        self._X = X
        self.peak_map = peak_map

        plot_obj_signal = None
        plot_obj_maxima = None
        plot_obj_whh = None
        plot_obj_bases = None

        peak_idx = (
            self.peak_map.maxima.pipe(pl.from_pandas)
            .select("p_idx")
            .unique(maintain_order=True)
            .to_pandas()
        )
        assert isinstance(peak_idx, pd.DataFrame)
        self.peak_color_map = assign_colors_to_p_idx(p_idx=peak_idx)

        # self.peak_map = join_peak_map_colors(peak_map, assign_colors_to_p_idx(peak_map))

        self.maxima_idx_key = "idx"
        self.maxima_key = "maxima"
        self.color_key = "color"

        self.handles: list[Line2D | Artist] = []

        # self._peak_plotter = PeakPlotter(ax=ax)

        if signal:
            plot_obj_signal = self.draw_signal()
        if maxima:
            plot_obj_maxima = self.draw_maxima()
        if whh:
            warnings.warn("whh plot not implemented")
            plot_obj_whh = None
        if base:
            plot_obj_bases = self._draw_base_edges(msnt=base)

        peak_map_plots = PeakMapViz(
            signal=plot_obj_signal,
            maxima=plot_obj_maxima,
            whh=plot_obj_whh,
            bases=plot_obj_bases,
        )

        peak_map_plots.overlay()

        return peak_map_plots

    def _draw_base_edges(self, msnt: str = "prom"):
        """
        Draw open-ended triangles running from the left base to the maxima to the right base for each peak, grouped by p_idx. Requires some data manipulation such that each line will be defined by the three points mentioned, each labeled by the p_idx.

        :msnt: The msnt to plot, :base: choose from "pb", "whh", "prom", "all", or "none". Default is "pb"
        """

        # a maxima value for each of the peak msnts
        contour_plotting_table = self._prepare_contour_plotting_table(base=msnt)

        # )  # fmt: skip
        base_edge_plot_obj = contour_plotting_table.plot(
            x="idx", y="X", by=["p_idx", "msnt"], label="base_edges", line_dash="dotted"
        )
        return base_edge_plot_obj

    def _prepare_contour_plotting_table(self, base: str):
        """
        Prepare a square (ish) table of each of the contour measurements 'msnt' with sides 'side' and maxima, with y (X) and x (X_idx) columns ready for plotting.

        The primary use is to draw lines reaching from the maxima of each peak down to the contour line sides in order to get a feel for how the algorithms are mapping the signal.

        :base: choose from "pb", "whh", "prom", "all", or "none".
        """

        # assemble a maxima entry for each contour measurement of each peak. This will be appended to 'contour_line_bounds'.

        if base in ["pb", "whh", "prom"]:
            filtered_bounds = self.peak_map.contour_line_bounds.pipe(
                pl.from_pandas
            ).filter(pl.col(mp_defs.KEY_MSNT).eq(base))

        if base == "all":
            filtered_bounds = self.peak_map.contour_line_bounds

        if base == "none":
            return

        contour_bound_maximas = (
            filtered_bounds.select(pl.col([mp_defs.P_IDX, mp_defs.KEY_MSNT]))
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

        # concat with filtered_bounds

        concatenated_contour_maximas = pl.concat(
            [
                filtered_bounds.with_columns(
                    pl.col(mp_defs.DIM).replace({mp_defs.KEY_IDX_ROUNDED: mp_defs.IDX})
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
                        mp_defs.IDX,
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
            x="idx",
            y="X",
            label="X",
            title="X",
        )

        return plot_obj

    def draw_maxima(self):
        plot_obj = (
            self.peak_map.maxima.pipe(pl.from_pandas)
            .pivot(columns="dim", index="p_idx", values="value")
            .plot(x="idx", y="X", label="maxima", kind="scatter", by="p_idx")
        )

        return plot_obj

    @deprecation.deprecated
    def __draw_maxima_matplotlib(self) -> Self:
        """
        call `PeakPlotter.plot_maxima` to draw the maximas on the internal Axes. Needs to first subset the peak_map to `maxima_x_y`.

        :return: _description_
        :rtype: Self
        """
        label = "maxima"
        color_key = "color"
        peak_map_pl = pl.from_pandas(self.peak_map)

        __draw_annotated_maxima(
            peak_map=self.peak_map,
            ax=self.ax,
            x_key=self.maxima_idx_key,
            y_key=self.maxima_key,
            color_colname=self.color_key,
        )

        self.__add_proxy_artist_to_handles(
            label=label,
            line_2d=self.ax.lines[-1],
            color=__get_first_value(peak_map_pl, color_key),
        )

        return self

    @deprecation.deprecated
    def __draw_base_edges(
        self,
    ) -> Self:
        """
        Draw lines connecting the left and right bases of each peak to demonstrate the space allocated to each peak.
        """
        label = "base maxima interpol."

        peak_map_pl = pl.from_pandas(self.peak_map)
        color_key = "color"

        __draw_peak_base_edges(
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
            color=__get_first_value(peak_map_pl, color_key),
        )

        return self

    @deprecation.deprecated
    def __draw_base_vertices(
        self,
    ) -> Self:
        color_key = "color"
        label = "bases"
        peak_map_pl: pl.DataFrame = pl.from_pandas(self.peak_map)

        __draw_width_vertices(
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
            color=__get_first_value(peak_map_pl, color_key),
        )

        return self

    @deprecation.deprecated
    def __show(self) -> Self:
        """
        display the current Axes.

        Graphics(?) are drawn onto the internal Axes object by other methods within this class. Calling this method uses up the current Axes.
        """

        __add_handles_to_legend(ax=self.ax, handles=self.handles)
        plt.show()

        return self

    @deprecation.deprecated
    def __add_proxy_artist_to_handles(
        self,
        label: str,
        line_2d: Line2D,
        color,
    ) -> None:
        """
        Code lifted directly from matplotlib legend guide <https://matplotlib.org/stable/users/explain/axes/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists>
        """

        legend_proxy = __set_legend_proxy_artist(label, line_2d, color)

        self.handles += [legend_proxy]
