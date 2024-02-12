from dataclasses import dataclass
from typing import Literal, Optional, Self, TypeAlias

import numpy as np
import pandas as pd
import pandera as pa
import polars as pl
from matplotlib.axes import Axes as Axes
from numpy import float64, int64
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series
from scipy.ndimage import label  # type: ignore

from hplc_py.hplc_py_typing.hplc_py_typing import (
    PeakWindows,
    SignalDFBCorr,
    WindowedSignal,
    X_Schema,
    X_PeakWindowed,
    X_Windowed,
)
from hplc_py.hplc_py_typing.typed_dicts import FindPeaksKwargs
from hplc_py.io_validation import IOValid
from hplc_py.map_signals.map_peaks.map_peaks import PeakMapWide
from hplc_py.pandera_helpers import PanderaSchemaMethods

from .map_peaks.map_peaks import MapPeaks


class MapWindows(IOValid):
    def __init__(
        self,
        prominence: float = 0.01,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ):
        self._w_type_colname = "w_type"
        self._w_idx_colname = "w_idx"
        self._X_colname = "X"
        self._X_idx_colname = "X_idx"

        self.w_type_interpeak_label: int = -999
        self.mp = MapPeaks(
            prominence=prominence, wlen=wlen, find_peaks_kwargs=find_peaks_kwargs
        )

        self.X_w = pl.DataFrame()

    def fit(self, X: DataFrame[X_Schema], timestep: float, y=None) -> Self:

        if isinstance(X, DataFrame):
            self.X = pl.from_pandas(X)
        else:
            self.X = X

        self.timestep = timestep
        self.mp.fit(X=X)

        return self

    def transform(
        self,
    ) -> Self:
        """
        Assign a peak and nonpeak window mapping to the signal based on peak overlap.
        Access self.X_w to retreive the windowed signal.

        :return: self
        :rtype: Self
        """
        self.mp.transform()

        left_bases = self.mp.peak_map.pb_left
        right_bases = self.mp.peak_map.pb_right

        peak_wdw_intvls = self._p_wdw_intvl_factory(left_bases, right_bases)
        self.X_w: DataFrame[X_Windowed] = self._window_X(self.X, peak_wdw_intvls)

        return self

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

    def _peak_base_intvl_factory(
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

    def _map_wdws_to_peaks(
        self,
        intvls: Series[pd.Interval],
    ) -> dict[int, list[int]]:
        """
        Takes a Series of pd.Interval objects representing the mapped base width of each
        peak and maps them to windows based on whether peaks overlap. Returns a dict of
        the mapping as w_idx: list of peak_idx.

        Note: the peak idxs are not in linear order, but rather left base time order.
        The window assignment logic requires sorting the peak intervals by the time
        of the left base, the order of which can differ from the peak_idx, because the
        peak idx is assigned based on location of the maxima, rather than the the peak
        base widths, which can be much further away, hence overlapping.

        :param intvls: a pd.Series of pd.Interval objects of peak base widths.
        :type intvls: Series[pd.Interval]
        :returns: a dict whose keys are window_idx and whose values are lists of peak
        indices mapped to the window.
        :type returns: dict[int, list[int]]
        """

        # checks
        self._is_a_series(intvls)
        self._is_an_interval_series(intvls)
        self._is_not_empty(intvls)

        # group first then form the intervals second. use 'index' for the group values

        w_idx = 0

        wdw_peak_mapping = {}

        """        
        Works as follows:
        1. take the interval objects and sort them. Sort behavior is defined by the __lt__
        method, which in this case is defined on whether the left bound is greater or smaller.
        For each iteration of i:
        1. compare ith interval with the current w_idx interval.
            1. if overlap:
                1. set the current w_idx interval to include the rhs of the ith
            interval.
                2. assign the current interval idx (i) to the current w_idx.
                3. move to next iteration of i.
            2. if not overlap:
                1. move w_idx up by one.
                2. assign current i to new w_idx in mapping dict
                3. set current w_idx bound.
                4. move to next iteration of i.      
        """

        #  we need to sort the intervals by time in order to check for overlap in a left
        # to right fashion, otherwise locally non-overlapping peaks can disrupt the assignment.
        # this way we compensate for later peaks whose overlap is not detected until later

        Intvls = sorted(
            [Interval(intvl.left, intvl.right, i) for i, intvl in enumerate(intvls)]
        )

        # setup the indices and default values
        w_idx = 0
        i_idx = np.arange(1, len(intvls))
        wdw_peak_mapping = {0: [Intvls[0].p_idx]}
        w_idx_Intvl = Intvls[0]

        # The core logic. While iterating over the intervals, first check if the current
        # window idx overlaps with the current peak idx. if so, assign that peak idx
        # to the window and extend the window interval by that peak interval. If not,
        # move to the next window idx, set the current peak interval as the current window
        # interval, and assign the peak idx as the first peak idx of the current window idx in the mapping.

        # note: could directly assign the window intervals to window idxs here, but in the
        # interest of preserving transparency, I have chosen to separate that functionality.

        for i in i_idx:
            does_overlap = w_idx_Intvl.overlaps(Intvls[i])
            if does_overlap:
                w_idx_Intvl += Intvls[i]
                wdw_peak_mapping[w_idx].append(Intvls[i].p_idx)
            else:
                w_idx += 1
                wdw_peak_mapping[w_idx] = [Intvls[i].p_idx]
                w_idx_Intvl = Intvls[i]

        return wdw_peak_mapping

    def _set_wdw_intvls_from_peak_intvls(
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

        # w_idxs is a dict whose keys are w_idxs and values are lists of p_idx belonging
        # to the w_idx. This function iterates through the dictionary and constructs
        # a pd.Interval object that represents each peak window.

        # if there is only 1 peak in the window, the window interval is equal to the
        # peak interval, else if the length is greater than one, then the window interval
        # is constructed from the first peak intervals left value, and the last peak
        # intervals right value.

        # the constructed interval is then assigned to the w_intvls dict as the value
        # whose key is the current w_idx.
        for w_idx, wdws in w_idxs.items():
            if len(wdws) == 1:
                w_intvls[w_idx] = intvls[wdws[0]]
            elif len(wdws) > 1:
                w_intvl = pd.Interval(
                    intvls[wdws[0]].left, intvls[wdws[-1]].right, "both"
                )
                w_intvls[w_idx] = w_intvl

            else:
                raise ValueError("expected a value for every window")

        return w_intvls

    @pa.check_types
    def _peak_intvls_as_frame(
        self,
        peak_wdw_intvls: dict[int, pd.Interval],
    ) -> DataFrame[PeakWindows]:
        """
        Given a dict of Interval objects, subset the X series idx by the bounds of the
        Interval, then label the subset with the w_idx, also labeling window type.
        """
        """
        Iterate through the interval dict and assemble an w_idx column with corresponding w_type column. 
        """
        p_wdw_list = []

        for i, intvl in peak_wdw_intvls.items():
            p_wdw = self.intvl_as_frame(i, intvl)
            p_wdw_list.append(p_wdw)

        p_wdws_ = pd.concat(p_wdw_list).reset_index(drop=True).rename_axis(index="idx")

        p_wdws = DataFrame[PeakWindows](p_wdws_)

        return p_wdws

    def intvl_as_frame(self, i, intvl):
        X_idx: NDArray[int64] = np.arange(intvl.left, intvl.right, 1)

        p_wdw: pd.DataFrame = pd.DataFrame(
            {
                self._X_idx_colname: X_idx,
            }
        )

        p_wdw[self._w_type_colname] = "peak"
        p_wdw[self._w_idx_colname] = i

        p_wdw = p_wdw[[self._w_type_colname, self._w_idx_colname, self._X_idx_colname]]

        p_wdw = p_wdw.astype(
            {
                self._w_idx_colname: int64,
                self._w_type_colname: pd.StringDtype(),
                self._w_idx_colname: int64,
            }
        )
        return p_wdw

    @pa.check_types
    def _set_peak_wndwd_X_idx(
        self,
        X: DataFrame[X_Schema],
        X_idx_pw: DataFrame[PeakWindows],
    ) -> DataFrame[X_PeakWindowed]:
        """
        Broadcasts the identified peak windows to the length of the series X idx axis.
        Any time without an assigned window is labeled as `self.w_type_interpeak_label`,
        i.e. -999 to indicate 'missing'. The patterns of missing values are later computed
        to label the interpeak windows.
        """
        if isinstance(X, DataFrame):
            X = pl.from_pandas(X)
        # joins the peak window intervals to the time index to generate a pattern of missing values

        p_wdws_ = pl.from_pandas(X_idx_pw, schema_overrides={"X_idx": pl.UInt32()})

        # left join time idx to peak windows and peak types, leaving na's to be filled
        # with a placeholder and 'interpeak', respectively.

        X_pw_broadcast = (
            X.with_row_index(self._X_idx_colname)
            .join(p_wdws_, on=self._X_idx_colname, how="left")
            .drop(self._X_idx_colname)
        )
        # not here

        # this simply fills the nulls
        x_w_ = X_pw_broadcast.with_columns(
            **{
                self._w_type_colname: pl.col(self._w_type_colname).fill_null(
                    "interpeak"
                ),
                self._w_idx_colname: pl.col(self._w_idx_colname).fill_null(
                    self.w_type_interpeak_label
                ),
            }
        ).select(self._w_type_colname, self._w_idx_colname, self._X_colname)

        x_w_pd_ = X_PeakWindowed.validate(
            x_w_.to_pandas().rename_axis(index="idx"), lazy=True
        )

        X_w_ = DataFrame[X_PeakWindowed](x_w_pd_)
        return X_w_

    def _get_interpeak_w_idxs(
        self,
        pwdwd_time: DataFrame[X_PeakWindowed],
    ) -> NDArray[int64]:
        labels = []
        labels, num_features = label(
            pwdwd_time[self._w_idx_colname] == self.w_type_interpeak_label
        )  # type: ignore
        labels = np.asarray(labels, dtype=np.int64)
        return labels

    # @pa.check_types
    def _label_interpeaks(
        self,
        X_pw: DataFrame[X_PeakWindowed],
    ) -> DataFrame[WindowedSignal]:
        """
        Simply replaces the interpeak placeholder index with the mapping obtained from
        "get_na_labels"
        """

        X_w_: pd.DataFrame = X_pw.copy(deep=True)

        labels = self._get_interpeak_w_idxs(X_pw)

        X_w_[self._w_idx_colname] = X_w_[self._w_idx_colname].mask(
            X_w_[self._w_idx_colname] == self.w_type_interpeak_label,
            Series(labels - 1),
        )

        X_w_ = X_w_.rename_axis(index="idx")

        X_Windowed.validate(X_w_, lazy=True)

        X_w: DataFrame[X_Windowed] = DataFrame[X_Windowed](X_w_)

        return X_w

    def _window_X(
        self, X: DataFrame[X_Schema], peak_wdw_intvls: dict[int, pd.Interval]
    ) -> DataFrame[X_Windowed]:
        """
        Window X by broadcasting the identified peak window intervals to the length
        of the signal. Patterns in missing window labels are coded as interpeak with
        their own ascending index.
        """

        # X_idw_w is the signal indices derived from the interval objects, rather than
        # directly from X.
        X_idx_pw: DataFrame[PeakWindows] = self._peak_intvls_as_frame(peak_wdw_intvls)
        # X_pw is the result of joining X with X_idx_w, with an arbitrary label where
        # interpeak windows are present
        X_pw: DataFrame[X_PeakWindowed] = self._set_peak_wndwd_X_idx(X, X_idx_pw)

        X_w: DataFrame[X_Windowed] = self._label_interpeaks(X_pw)

        return X_w

    def _p_wdw_intvl_factory(self, left_bases, right_bases):
        left_bases_, right_bases_ = self._check_bases(left_bases, right_bases)

        # get the peak bases as pd.Intveral objects, then map the peaks to windows
        # depending on whether they overlap
        # then construct the window interval objects.
        pb_intvls = self._peak_base_intvl_factory(left_bases_, right_bases_)
        wdw_peak_mapping: dict[int, list[int]] = self._map_wdws_to_peaks(pb_intvls)
        peak_wdw_intvls: dict[int, pd.Interval] = self._set_wdw_intvls_from_peak_intvls(
            pb_intvls, wdw_peak_mapping
        )
        return peak_wdw_intvls

    def _check_bases(self, left_bases, right_bases):
        left_bases_: Series[int64] = Series(left_bases, dtype=int64)
        right_bases_: Series[int64] = Series(right_bases, dtype=int64)

        # sanity check: all left bases should be less than corresponding right base

        base_check = pd.concat([left_bases_, right_bases_], axis=1)
        # horizontal less

        base_check = base_check.assign(hz_less=lambda df: df.iloc[:, 0] < df.iloc[:, 1])
        if not base_check["hz_less"].all():
            raise ValueError(
                f"expect left_base of each pair to be less than corresponding right.\n\n{base_check}"
            )

        return left_bases_, right_bases_


class Interval:
    """
    A simplistic class to handle comparision and addition of intervals. Assumes
    closed intervals.
    """

    def __init__(self, left: int, right: int, p_idx: int):
        """
        assign the left, right and p_idx of the Interval object. the p_idx keeps track
        of which peak this inteval belongs to. The left and right are the closed bounds
        of the interval.

        :param left: the left closed bound of the interval
        :type left: int
        :param right: the right closed bound of the interval
        :type right: int
        :param p_idx: the index of the peak the interval belongs to
        :type p_idx: int
        :param self: an interval object with comparision and extension through `__add__`
        :type self: Interval
        """
        self.left = left
        self.right = right
        self.p_idx = p_idx

    def overlaps(
        self,
        other,
    ):
        """
        Assess if two Interval abjects overlap by compring their left and right
        bounds. Assumes closed intervals.
                    i1   i2                 i2   i1
        case 1: |----||--|-----||  case 2: ||---|---||----|

        in either case, sorting the interval objects so that we're always in the
        frame of reference of case 1 then comparing i1 to i2. If right i1 is
        greater than or equal to the left of i2, then they are deemed overlapping.
        """

        if not isinstance(other, Interval):
            raise TypeError("can only compare Interval objects")

        i1, i2 = sorted([self, other])

        if i1.right >= i2.left:
            return True
        else:
            return False

    def __add__(self, other):
        if self.left < other.left:
            new_left = self.left
        else:
            new_left = other.left

        if self.right < other.right:
            new_right = other.right
        else:
            new_right = self.right

        return Interval(new_left, new_right, p_idx=self.p_idx)

    def __lt__(self, other):
        return self.left < other.left

    def __repr__(self):
        return f"{self.p_idx}: [{self.left}, {self.right}]"
