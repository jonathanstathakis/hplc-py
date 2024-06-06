import holoviews as hv
from .map_peaks.viz_hv import PeakMapViz
from .map_signal import SignalMap
from .map_windows.viz_hv import WindowMapViz


class SignalMapViz:
    def __init__(self, signal_map: SignalMap):
        """
        mimics the API design of PeakMapViz / WindowMapViz, i.e. method chains before
        collection, recreating the methods and collecting the plots at this level.
        """
        self.signal_map = signal_map
        self.peaks: PeakMapViz = self.signal_map.peak_map.as_viz()
        self.windows: WindowMapViz = self.signal_map.window_map.as_viz()

        self.plots: dict = {}

    def draw_signal(self):

        plot_obj = self.peaks.draw_signal().plot()

        self.plots["signal"] = plot_obj

        return self

    def draw_maxima(self):

        plot_obj = self.peaks.draw_maxima().plot()

        self.plots["maxima"] = plot_obj

        return self

    def draw_base_edges(
        self, msnt: str = "pb", plot_kwargs: dict = dict(line_dash="dotted")
    ):
        plot_obj = self.peaks.draw_base_edges().plot()

        self.plots["base_edges"] = plot_obj

    def draw_interpeak_windows(self):

        plot_obj = self.windows.draw_interpeak_windows().plot()

        self.plots["interpeak_windows"] = plot_obj

        return self

    def label_peak_windows(self):
        plot_obj = self.windows.label_peak_windows().plot()

        self.plots["peak_window_labels"] = plot_obj

        return self

    def plot(self):

        plot_obj = hv.Overlay(self.plots.values())

        self.plots.clear()

        return plot_obj
