from hplc_py.map_signal.map_windows import window_map_output
import holoviews as hv
import polars as pl


class WindowMapViz:
    """
    collect plot objects into the plots dict via method chain, produce overlay and output with 'plots'
    """

    def __init__(self, window_map):
        self.height = 500
        self.width = 1000
        self.window_map = window_map
        self.plots = {}

    def draw_interpeak_windows(self):
        x_start = self.window_map.interpeak_window_left_bounds
        x_end = self.window_map.interpeak_window_right_bounds
        import holoviews as hv

        spans = hv.VSpans((x_start, x_end)).opts(height=self.height, width=self.width)

        self.plots["interpeak_windows"] = spans

        return self

    def label_peak_windows(self):

        x = self.window_map.peak_windows_long.group_by(
            "w_idx", maintain_order=True
        ).agg(pl.col("idx").mean().alias("x"))

        y = self.window_map.X_maxima

        draw_data = x.with_columns(pl.lit(y).alias("y"))

        labels = draw_data.plot.labels(x="x", y="y", text="w_idx")

        self.plots["labels"] = labels
        return self

    def draw_signal(self):

        signal_plot = self.window_map.X_windowed.plot(x="idx", y="X")
        self.plots["signal"] = signal_plot
        return self

    def plot(self):
        plot = hv.Overlay(self.plots.values())
        return plot
