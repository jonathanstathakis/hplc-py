from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.axes import Axes as Axes
import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from scipy.ndimage import label  # type: ignore

from hplc_py.hplc_py_typing.hplc_py_typing import (
    IntArray,
    SignalDF,
    BaseDF,
)
from hplc_py.map_signals.map_peaks import PeakMap
from typing import Type

from hplc_py import AMPRAW, AMPCORR

class WindowedSignal(BaseDF):
    window_type: pd.StringDtype
    window_idx: pd.Int64Dtype
    time_idx: pd.Int64Dtype
    time: pd.Float64Dtype
    amp: pd.Float64Dtype=pa.Field(regex=f"\b({AMPRAW}|{AMPCORR})\b")
    

class PeakWindows(BaseDF):
    time_idx: pd.Int64Dtype
    time: pd.Float64Dtype
    window_idx: pd.Int64Dtype
    window_type: pd.StringDtype

    class Config:
        strict = True


class IPBounds(BaseDF):
    ip_window_idx: pd.Int64Dtype
    ip_bound: pd.StringDtype
    time_idx: pd.Int64Dtype
    ip_window_type: pd.StringDtype

    @pa.check("ip_window_idx", name="check_w_idx_increasing")
    def check_monotonic_increasing_w_idx(cls, s: Series[pd.Int64Dtype]) -> bool:
        return s.is_monotonic_increasing

    @pa.check("time_idx", name="check_time_idx_increasing")
    def check_monotonic_increasing_t_idx(cls, s: Series[pd.Int64Dtype]) -> bool:
        return s.is_monotonic_increasing

    class Config:
        strict = True


class PWdwdTime(BaseDF):
    """
    peak windowed time dataframe, with NA's for nonpeak regions. An intermediate frame prior to full mapping
    """

    time_idx: pd.Int64Dtype
    time: pd.Float64Dtype
    window_idx: pd.Int64Dtype = pa.Field(nullable=True)
    window_type: pd.StringDtype = pa.Field(nullable=True)

    class Config:
        strict = True


class WindowedTime(PWdwdTime):
    window_idx: pd.Int64Dtype
    window_type: pd.StringDtype


@dataclass
class MapWindowsMixin:
    pm = PeakMap
    sdf = SignalDF
    pw_sc = PeakWindows
    ipb_sc = IPBounds
    pwdt_sc = PWdwdTime
    wt_sc = WindowedTime
    ws_sc = WindowedSignal

    """
    - create the intervals
    - note which intervals overlap
        - create a table of intervals
        - loop through from first to last in time order and check if they overlap with `.overlaps`
            - if no overlap, move to next interval
    - combine the intervals if they overlap
    - mask the time index with the overlap index, unmasked intervals labeled as 'nonpeak'
    outcome: two level index: window_type, window_num. window_type has two vals: 'peak', 'nonpeak'
    """

    def _is_not_empty(
        self,
        ndframe: pd.DataFrame | pd.Series,
    ) -> None:
        if ndframe.empty:
            raise ValueError(f"{ndframe.name} is empty")

    def _is_a_series(
        self,
        s: pd.Series,
    ) -> None:
        if not isinstance(s, pd.Series):
            if s.empty:
                raise ValueError(f"{s.name} is empty")

    def _is_an_int_series(
        self,
        s: pd.Series,
    ) -> None:
        if not pd.api.types.is_integer_dtype(s):
            raise TypeError(f"Expected {s.name} to be int dtype, got {s.dtype}")

    def _is_an_interval_series(
        self,
        s: pd.Series,
    ) -> None:
        if not isinstance(s.dtype, pd.IntervalDtype):
            raise TypeError(f"Expected {s.name} to be Interval dtype, got {s.dtype}")

    def _interval_factory(
        self,
        left_bases: Series[pd.Int64Dtype],
        right_bases: Series[pd.Int64Dtype],
    ) -> Series[pd.Interval]:
        """
        Take the left an right base index arrays and convert them to pandas Intervals, returned as a frame. Assumes that the left and right base arrays are aligned in order
        """
        # checks

        # check that s is a series
        # left and right base are int arrays
        for s in [left_bases, right_bases]:
            self._is_a_series(s)
            self._is_an_int_series(s)
            self._is_not_empty(s)

        # to convert the left and right bases to series, need to zip them together then use each pair as the closed bounds.

        intvls: pd.Series[pd.Interval] = pd.Series(name="peak_intervals")

        for i, pair in enumerate(zip(left_bases, right_bases)):
            intvls[i] = pd.Interval(pair[0], pair[1], closed="both")

        return intvls  # type: ignore

    def _label_windows(
        self,
        intvls: Series[pd.Interval],
    ) -> dict[int, list[int]]:
        """
        For a Series of closedpd.Interval objects, sort them into windows based on whether they overlap. Returns a dict of window_idx: interval list. for later combination.

        iterating through the intervals starting at zero, compare to the set of values
        already assigned to window indexes (initally none), if the interval index is already
        present, continue to the next interval, else set up a new window_idx list, and
        if the i+1th interval overlaps with interval i, assign it to the current window_idx.
        if no more intervals overlap, move to the next window index and the next interval iteration.
        """

        # checks
        self._is_a_series(intvls)
        self._is_an_interval_series(intvls)
        self._is_not_empty(intvls)

        # group first then form the intervals second. use 'index' for the group values

        w_idx = 0

        w_idxs = {}

        """

        """
        for i in range(0, len(intvls)):
            # check current
            mapped_vals = {x for v in w_idxs.values() for x in v}

            if i in mapped_vals:
                continue
            w_idxs[w_idx] = [i]
            for j in range(i + 1, len(intvls)):
                if intvls[i].overlaps(intvls[j]):
                    w_idxs[i].append(j)
            w_idx += 1

        return w_idxs

    def _combine_intvls(
        self,
        intvls: Series[pd.Interval],
        w_idxs: dict[int, list[int]],
    ) -> dict[int, pd.Interval]:
        """
        combine the intervals by window to cover the extent of the window. iterate through
        w_idxs, retrieve the intervals mapped to the idx, combine them and assign to a new dict

        Find the minimum and maximum of the grouped intervals and form a new interval from them. To this need to gather all the interval bounds together and compare them, left and right seperately
        """

        w_intvls = {}

        for i, length in w_idxs.items():
            if len(length) == 1:
                w_intvls[i] = intvls[length[0]]
            elif len(length) > 1:
                left_bounds = intvls[length].apply(lambda x: x.left)  # type: ignore
                right_bounds = intvls[length].apply(lambda x: x.right)  # type: ignore

                w_min = left_bounds.min()
                w_max = right_bounds.max()
                w_intvl = pd.Interval(w_min, w_max, "both")

                w_intvls[i] = w_intvl

            else:
                raise ValueError("expected a value for every window")

        return w_intvls

    def _set_peak_windows(
        self,
        w_intvls: dict[int, pd.Interval],
        time: Series[pd.Float64Dtype],
    ) -> DataFrame[PeakWindows]:
        """
        Given a dict of Interval objects, subset the time series by the bounds of the
        Interval, then label the subset with the window_idx, also labeling window type.
        """

        intvl_times = []

        for i, intvl in w_intvls.items():
            time_idx: IntArray = np.arange(intvl.left, intvl.right, 1)

            time_intvl: pd.DataFrame = pd.DataFrame(
                {
                    "time_idx": time_idx,
                    "time": time[intvl.left : intvl.right],
                }
            )

            time_intvl["window_idx"] = i
            time_intvl["window_type"] = "peak"

            time_intvl = time_intvl.astype(
                {
                    "time_idx": pd.Int64Dtype(),
                    "time": pd.Float64Dtype(),
                    "window_idx": pd.Int64Dtype(),
                    "window_type": pd.StringDtype(),
                }
            )

            intvl_times.append(time_intvl)

        intvl_df_ = (
            pd.concat(intvl_times).reset_index(drop=True).rename_axis(index="idx")
        )

        intvl_df = DataFrame[PeakWindows](intvl_df_)

        return intvl_df

    def _set_peak_wndwd_time(
        self,
        time: Series[pd.Float64Dtype],
        peak_wdws: DataFrame[PeakWindows],
    ) -> DataFrame[PWdwdTime]:
        if time.dtype != pd.Float64Dtype():
            raise TypeError(f"expected time to be NDArray of float, got {time.dtype}")
        # constructs a time dataframe with time value and index
        wdwd_time: pd.DataFrame = pd.DataFrame(
            {"time_idx": np.arange(0, len(time), 1), "time": time}
        )

        # joins the peak window intervals to the time index to generate a pattern of missing values
        wdwd_time = (
            wdwd_time.set_index("time_idx")
            .join(peak_wdws.set_index("time_idx").drop("time", axis=1), how="left")
            .reset_index()
            .rename_axis(index="idx")
        )

        # label the peak windows
        wdwd_time["window_type"] = wdwd_time["window_type"].where(
            wdwd_time["window_type"] == "peak", np.nan
        )
        try:
            wdwd_time = DataFrame[PWdwdTime](wdwd_time)
        except pa.errors.SchemaErrors as e:
            raise e

        return wdwd_time

    def _get_na_labels(
        self,
        pwdwd_time: DataFrame[PWdwdTime],
        w_idx_col: str,
    ) -> IntArray:
        labels = []

        labels, num_features = label(pwdwd_time[w_idx_col].isna())  # type: ignore
        labels = np.asarray(labels, dtype=np.int64)
        return labels

    def _label_interpeaks(
        self,
        pwdwd_time: DataFrame[PWdwdTime],
        w_idx_col: str,
    ) -> DataFrame[WindowedTime]:
        labels = self._get_na_labels(pwdwd_time, w_idx_col)

        pwdwd_time["window_idx"] = pwdwd_time["window_idx"].mask(
            pwdwd_time["window_idx"].isna(), Series(labels - 1)
        )

        pwdwd_time["window_type"] = pwdwd_time["window_type"].mask(
            pwdwd_time["window_type"].isna(), "interpeak"
        )

        wdwd_time: DataFrame[WindowedTime] = DataFrame[WindowedTime](
            pwdwd_time.copy(deep=True)
        )

        return wdwd_time

    def _join_sdf_wm(
        self,
        amp: Series[pd.Float64Dtype],
        wt: DataFrame[WindowedTime],
    ) -> DataFrame[WindowedSignal]:
        
        ws = wt.copy(deep=True)
        
        ws['amp']=Series[pd.Float64Dtype](amp, dtype=pd.Float64Dtype())

        ws = ws.set_index(['window_type','window_idx','time_idx','time',]).reset_index().rename_axis(index='idx')
        
        try:
            ws = DataFrame[WindowedSignal](ws)
        except pa.errors.SchemaError as e:
            e.add_note(str(ws))
            raise e

        return ws

    def _map_windows(
        self,
        left_bases: Series[pd.Float64Dtype],
        right_bases: Series[pd.Float64Dtype],
        time: Series[pd.Float64Dtype],
    ) -> DataFrame[WindowedTime]:
        # convert left and right_bases to int. At this point assuming that all ceiling will be acceptable

        left_bases_: Series[pd.Int64Dtype] = Series(left_bases, dtype=pd.Int64Dtype())
        right_bases_: Series[pd.Int64Dtype] = Series(right_bases, dtype=pd.Int64Dtype())

        intvls = self._interval_factory(left_bases_, right_bases_)

        w_idxs: dict[int, list[int]] = self._label_windows(intvls)
        w_intvls: dict[int, pd.Interval] = self._combine_intvls(intvls, w_idxs)
        pw: DataFrame[PeakWindows] = self._set_peak_windows(w_intvls, time)
        pwt: DataFrame[PWdwdTime] = self._set_peak_wndwd_time(time, pw)
        wt: DataFrame[WindowedTime] = self._label_interpeaks(
            pwt, str(self.pwdt_sc.window_idx)
        )

        return wt


from hplc_py.map_signals.map_peaks import MapPeakPlots


@dataclass
class MapWindowPlots(MapPeakPlots):
    ws_sch: Type[WindowedSignal] = WindowedSignal

    def __post_init__(self):
        super().__init__()
    
    def _rectangle_factory(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        angle: float=0.0,
        rotation_point: str = 'xy',
        rectangle_kwargs={},
    )->Rectangle:
        
        rectangle = Rectangle(xy, width, height, angle=angle, rotation_point=rotation_point, **rectangle_kwargs)
        
        return rectangle
    
    def plot_windows(
        self,
        ws: DataFrame[WindowedSignal],
        height: float,
        ax: Axes = None,
        rectangle_kwargs: dict={},
    ):  
        """
        Plot each window as a Rectangle
        
        height is the maxima of the signal.
        """
        
        if not ax:
            ax=plt.gca()
        
        # rectangle definition: class matplotlib.patches.Rectangle(xy, width, height, *, angle=0.0, rotation_point='xy', **kwargs)
        # rectangle usage: `ax.add_collection([Rectangles..])` or `ax.add_patch(Rectangle)``
        
        
        
        window_stats = ws.groupby([self.ws_sch.window_type, self.ws_sch.window_idx])[self.ws_sch.time].agg(['min','max'])
        
        rh = height*1.05
        
        # peak windows
        rectangles = []
        for k, g in window_stats.groupby([self.ws_sch.window_type,self.ws_sch.window_idx]):
            x0 = g['min'].iat[0]
            y0 = 0
            width = g['max'].iat[0]-x0
            
            rectangle = self._rectangle_factory(
                (x0, y0),
                width,
                rh,
                rectangle_kwargs=rectangle_kwargs,
            )
        
            rectangles.append(rectangle)        
        
        import matplotlib as mpl
        
        cmap = mpl.colormaps["Set1"].resampled(len(window_stats))
        
        pc = PatchCollection(
            rectangles,
            zorder=0,
            alpha=0.25,
            facecolors=cmap.colors,
            )
        ax.add_collection(pc)
        
        pass


@dataclass
class MapWindows(MapWindowsMixin):
    def window_signal(
        self,
        left_bases: Series[pd.Float64Dtype],
        right_bases: Series[pd.Float64Dtype],
        time: Series[pd.Float64Dtype],
        amp: Series[pd.Float64Dtype],
    ) -> DataFrame[WindowedSignal]:

        wt = self._map_windows(
            left_bases,
            right_bases,
            time,
        )

        wsdf = self._join_sdf_wm(
            amp,
            wt,
        )

        return wsdf
