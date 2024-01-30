from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera as pa
from matplotlib.axes import Axes as Axes
from numpy import float64, int64
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series
from scipy.ndimage import label  # type: ignore

from hplc_py.hplc_py_typing.hplc_py_typing import (
    PeakWindows,
    PWdwdTime,
    SignalDFBCorr,
    WindowedSignal,
    WindowedTime,
)
from hplc_py.io_validation import IOValid
from hplc_py.map_signals.map_peaks.map_peaks import PeakMap


@dataclass
class MapWindowsMixin(IOValid):
    pm_sc = PeakMap
    sdf_sc = SignalDFBCorr
    pw_sc = PeakWindows
    pwdt_sc = PWdwdTime
    wt_sc = WindowedTime
    ws_sc = WindowedSignal
    w_type_interpeak_label:int = -999

    """
    - create the intervals
    - note which intervals overlap
        - create a table of intervals
        - loop through from first to last in time order and check if they overlap with `.overlaps`
            - if no overlap, move to next interval
    - combine the intervals if they overlap
    - mask the time index with the overlap index, unmasked intervals labeled as 'nonpeak'
    outcome: two level index: w_type, window_num. w_type has two vals: 'peak', 'nonpeak'
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
        left_bases: Series[int64],
        right_bases: Series[int64],
    ) -> Series[pd.Interval]:
        """
        Take the left an right base index arrays and convert them to pandas Intervals, returned as a frame. Assumes that the left and right base arrays are aligned in order
        """
        # checks

        # check that s is a series
        # left and right base are int arrays
        for s in [left_bases, right_bases]:
            self.check_container_is_type(s, pd.Series, int64)

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
        For a Series of closedpd.Interval objects, sort them into windows based on whether they overlap. Returns a dict of w_idx: interval list. for later combination.

        iterating through the intervals starting at zero, compare to the set of values
        already assigned to window indexes (initally none), if the interval index is already
        present, continue to the next interval, else set up a new w_idx list, and
        if the i+1th interval overlaps with interval i, assign it to the current w_idx.
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
        time: Series[float64],
    ) -> DataFrame[PeakWindows]:
        """
        Given a dict of Interval objects, subset the time series by the bounds of the
        Interval, then label the subset with the w_idx, also labeling window type.
        """

        intvl_times = []

        for i, intvl in w_intvls.items():
            time_idx: NDArray[int64] = np.arange(intvl.left, intvl.right, 1)

            time_intvl: pd.DataFrame = pd.DataFrame(
                {
                    "time_idx": time_idx,
                    "time": time[intvl.left : intvl.right],
                }
            )

            time_intvl["w_idx"] = i
            time_intvl["w_type"] = "peak"

            time_intvl = time_intvl.astype(
                {
                    "time_idx": int64,
                    "time": float64,
                    "w_idx": int64,
                    "w_type": pd.StringDtype(),
                }
            )

            intvl_times.append(time_intvl)

        intvl_df_ = (
            pd.concat(intvl_times).reset_index(drop=True).rename_axis(index="idx")
        )

        intvl_df = DataFrame[PeakWindows](intvl_df_)

        return intvl_df
    
    @pa.check_types
    def _set_peak_wndwd_time(
        self,
        time: Series[float64],
        peak_wdws: DataFrame[PeakWindows],
    ) -> DataFrame[PWdwdTime]:
        
        self.check_container_is_type(time, pd.Series, float64)
        
        # constructs a time dataframe with time value and index
        
        wdwd_time_: pd.DataFrame = pd.DataFrame(
            {"time_idx": np.arange(0, len(time), 1), "time": time}
        )

        # joins the peak window intervals to the time index to generate a pattern of missing values
        wdwd_time = (
            wdwd_time_.set_index("time_idx")
            .join(peak_wdws.set_index("time_idx").drop("time", axis=1), how="left")
            .reset_index()
            .rename_axis(index="idx")
            .assign(w_type=lambda df: df['w_type'].fillna('interpeak'))
            .assign(w_idx=lambda df: df['w_idx'].fillna(self.w_type_interpeak_label))
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
    ) -> NDArray[int64]:
        
        if not isinstance(w_idx_col, str):
            raise TypeError(f"expected str, got {type(w_idx_col)}")
        
        labels = []
        labels, num_features = label(pwdwd_time[w_idx_col]==self.w_type_interpeak_label)  # type: ignore
        labels = np.asarray(labels, dtype=np.int64)
        return labels

    @pa.check_types
    def _label_interpeaks(
        self,
        pwdwd_time: DataFrame[PWdwdTime],
        w_idx_col: str,
    ) -> DataFrame[WindowedTime]:
        
        pwdwd_time_ = pwdwd_time.copy(deep=True)
        
        self._check_scalar_is_type(w_idx_col, str)
        
        labels = self._get_na_labels(pwdwd_time, w_idx_col)

        pwdwd_time_["w_idx"] = pwdwd_time_["w_idx"].mask(
            pwdwd_time_["w_idx"]==self.w_type_interpeak_label, Series(labels - 1)
        )

        wdwd_time: DataFrame[WindowedTime] = DataFrame[WindowedTime](
            pwdwd_time_
        )

        return wdwd_time

    @pa.check_types
    def _join_ws_wt(
        self,
        amp: Series[float64],
        wt: DataFrame[WindowedTime],
    ) -> DataFrame[WindowedSignal]:
        
        self.check_container_is_type(amp, pd.Series, float64)
        
        ws = wt.copy(deep=True)

        ws["amp"] = Series[float64](amp, dtype=float64)

        ws = (
            ws.set_index(
                [
                    "w_type",
                    "w_idx",
                    "time_idx",
                    "time",
                ]
            )
            .reset_index()
            .rename_axis(index="idx")
        )
        try:
            ws = DataFrame[WindowedSignal](ws)
        except pa.errors.SchemaError as e:
            e.add_note(str(ws))
            raise e

        return ws

    @pa.check_types
    def _map_windows_to_time(
        self,
        left_bases: Series[float64],
        right_bases: Series[float64],
        time: Series[float64],
    ) -> DataFrame[WindowedTime]:
        """
        convert left and right_bases to int. At this point assuming that all ceiling will be acceptable
        """

        left_bases_: Series[int64] = Series(left_bases, dtype=int64)
        right_bases_: Series[int64] = Series(right_bases, dtype=int64)

        # sanity check: all left bases should be less than corresponding right base

        base_check = pd.concat([left_bases_, right_bases_], axis=1)
        # horizontal less

        base_check = base_check.assign(
            hz_less=lambda df: df.iloc[:,0] < df.iloc[:,1]
        )
        if not base_check["hz_less"].all():
            raise ValueError(
                f"expect left_base of each pair to be less than corresponding right.\n\n{base_check}"
            )

        intvls = self._interval_factory(left_bases_, right_bases_)
        w_idxs: dict[int, list[int]] = self._label_windows(intvls)
        w_intvls: dict[int, pd.Interval] = self._combine_intvls(intvls, w_idxs)
        pw: DataFrame[PeakWindows] = self._set_peak_windows(w_intvls, time)
        pwt: DataFrame[PWdwdTime] = self._set_peak_wndwd_time(time, pw)
        wt: DataFrame[WindowedTime] = self._label_interpeaks(
            pwt, str(self.pwdt_sc.w_idx)
        )

        return wt


@dataclass
class MapWindows(MapWindowsMixin):
    def window_signal(
        self,
        left_bases: Series[float64],
        right_bases: Series[float64],
        time: Series[float64],
        amp: Series[float64],
    ) -> DataFrame[WindowedSignal]:
        wt = self._map_windows_to_time(
            left_bases,
            right_bases,
            time,
        )

        wsdf = self._join_ws_wt(
            amp,
            wt,
        )

        return wsdf
