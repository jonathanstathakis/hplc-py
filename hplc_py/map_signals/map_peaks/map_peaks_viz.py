from matplotlib.lines import Line2D
import polars as pl
from typing import Self
from pandera.typing import DataFrame
from hplc_py.hplc_py_typing.hplc_py_typing import PeakMapWide, X_Schema

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist

from .peakplotfuncs import (
    plot_signal,
    assign_colors_to_p_idx,
    draw_annotated_maxima,
    join_peak_map_colors,
    draw_peak_base_edges,
    draw_width_vertices,
    set_legend_proxy_artist,
    add_handles_to_legend,
    get_first_value,
)


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
        label = 'maxima'
        color_key = 'color'
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
        label = 'base maxima interpol.'
        
        peak_map_pl = pl.from_pandas(self.peak_map)
        color_key = 'color'
        
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
