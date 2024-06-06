import copy
from sklearn.base import BaseEstimator, TransformerMixin
from .map_peaks import map_peaks
from .map_peaks import peak_map_output
from .map_windows import map_windows
from .map_windows.window_map_output import WindowMap
import logging

logger = logging.getLogger(__name__)




class SignalMapper(TransformerMixin, BaseEstimator):
    def __init__(self, find_peaks_kwargs: dict = {}):
        self.find_peaks_kwargs = find_peaks_kwargs
        self.peak_mapper = map_peaks.PeakMapper(find_peaks_kwargs=find_peaks_kwargs)
        self.window_mapper = map_windows.WindowMapper()

    def fit(self, X, y=None):
        self.X = X
        return self

    def transform(self, X, y=None):
        self.X = X

        self.peak_map: peak_map_output.PeakMap = X.pipe(self.peak_mapper.fit_transform)

        left_bases = self.peak_map.contour_line_bounds.get_base_side_as_series(
            side="left", msnt="pb", unit="idx_rounded"
        )
        right_bases = self.peak_map.contour_line_bounds.get_base_side_as_series(
            side="right", msnt="pb", unit="idx_rounded"
        )

        self.window_mapper.fit_transform(
            X=self.X, left_bases=left_bases, right_bases=right_bases
        )

        self.window_map: WindowMap = self.window_mapper.window_map

        self.signal_mapping_ = SignalMap(
            peak_map=self.peak_map, window_map=self.window_map
        )

        return self.signal_mapping_.window_map.X_windowed


class SignalMap:
    def __init__(
        self,
        peak_map: peak_map_output.PeakMap,
        window_map: WindowMap,
    ):
        self._peak_map: peak_map_output.PeakMap = peak_map
        self.window_map: WindowMap = window_map
        self.peak_map = self.windowed_peak_map()

    def viz_mode(self):
        from hplc_py.map_signal import viz_hv

        return viz_hv.SignalMapViz(self)

    def windowed_peak_map(self):
        """
        Use the windows assigned by WindowMapper to label the peak map data retroactively, replacing the output of PeakMapper with its windowed version.
        """

        # the window type, window index and time index from "window map". This is used to join to the "peak map" tables on the time idx
        time_window_mapping = self.window_map.time_window_mapping

        # The three peak map tables: "maxima", "contour line bounds" (bounds), and "widths" are extracted from the current peak map, windowed, and then passed to WindowedPeakMap constructor.

        windowed_maxima = (
            self._peak_map.maxima.pivot(
                index=["p_idx", "loc"], columns="dim", values="value"
            )
            .cast(dict(idx=int))
            .join(time_window_mapping, on="idx", how="left")
        )

        windowed_contour_line_bounds = (
            self._peak_map.contour_line_bounds.bounds.pivot(
                index=["p_idx", "loc", "msnt"], columns="dim", values="value"
            )
            .cast(dict(idx_rounded=int))
            .rename(dict(idx_rounded="idx"))
            .join(time_window_mapping, on="idx")
        )
        
        peak_idx_window_mapping = windowed_maxima.select(["p_idx", "w_type", "w_idx"])

        widths = self._peak_map.widths

        windowed_widths = widths.join(peak_idx_window_mapping, on="p_idx", how="left")

        windowed_peak_map = peak_map_output.WindowedPeakMap(
            maxima=windowed_maxima,
            contour_line_bounds=windowed_contour_line_bounds,
            widths=windowed_widths,
        )

        return windowed_peak_map

    def __repr__(self):
        return f"peak_map: {self._peak_map}\nwindow_map: {self.window_map}"
