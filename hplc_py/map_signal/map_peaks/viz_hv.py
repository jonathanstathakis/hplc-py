from hplc_py.map_signal.map_peaks import definitions as mp_defs
from hplc_py.map_signal.map_peaks import peak_map_output


import holoviews as hv
import polars as pl


class PeakMapViz:
    def __init__(self, peak_map):
        self.peak_map = peak_map
        self.plots = {}
        pass

    def draw_signal(self):
        """
        use hvplot namespace to produce a plot obj of the input X signal.
        """

        plot_obj = self.peak_map.X.plot(
            x="idx",
            y="X",
            label="X",
            title="X",
        )

        self.plots["signal"] = plot_obj
        return self

    def draw_maxima(self):

        plot_obj = self.peak_map.maxima.pivot(
            columns="dim", index="p_idx", values="value"
        ).plot(x="idx", y="X", label="maxima", kind="scatter", by="p_idx")

        self.plots["maxima"] = plot_obj

        return self

    def draw_base_edges(
        self, msnt: str = "pb", plot_kwargs: dict = dict(line_dash="dotted")
    ):
        """
        Draw open-ended triangles running from the left base to the maxima to the right base for each peak, grouped by p_idx. Requires some data manipulation such that each line will be defined by the three points mentioned, each labeled by the p_idx.

        :msnt: The msnt to plot, :base: choose from "pb", "whh", "prom", "all", or "none". Default is "pb"
        """

        # a maxima value for each of the peak msnts
        contour_plotting_table = self._prepare_contour_plotting_table(msnt=msnt)

        # )  # fmt: skip
        base_edge_plot_obj = contour_plotting_table.plot(
            x="idx", y="X", by=["p_idx", "msnt"], label="base_edges", **plot_kwargs
        )

        self.plots[msnt] = base_edge_plot_obj
        return self

    def _prepare_contour_plotting_table(self, msnt: str):
        """
        Prepare a square (ish) table of each of the contour measurements 'msnt' with sides 'side' and maxima, with y (X) and x (X_idx) columns ready for plotting.

        The primary use is to draw lines reaching from the maxima of each peak down to the contour line sides in order to get a feel for how the algorithms are mapping the signal.

        :base: choose from "pb", "whh", "prom", "all", or "none".
        """

        bounds = self.peak_map.contour_line_bounds.get_bound_by_type(msnt=msnt)

        # assemble a maxima entry for each contour measurement of each peak. This will be appended to 'contour_line_bounds'.

        contour_bound_maximas = (
            bounds.select(pl.col([mp_defs.P_IDX, mp_defs.KEY_MSNT]))
            .unique()
            .join(
                how="left",
                other=self.peak_map.maxima,
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
                bounds.with_columns(
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

    def plot(self) -> hv.Overlay:
        """
        collects all the plot objects into an overlay, clearing the store in the process
        """
        if not self.plots:
            raise RuntimeError("call a 'draw' method first")

        peak_map_overlay = hv.Overlay(self.plots.values())

        self.plots.clear()

        return peak_map_overlay
