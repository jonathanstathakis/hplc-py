from typing import Optional, Self

import numpy as np
import pandas as pd
import pandera as pa
import polars as pl
from matplotlib.axes import Axes as Axes
from numpy import int64
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series
from scipy.ndimage import label  # type: ignore

from hplc_py.hplc_py_typing.hplc_py_typing import (
    PeakWindows,
    WindowedSignal,
    X_Schema,
    X_PeakWindowed,
    X_Windowed,
)
from hplc_py.hplc_py_typing.typed_dicts import FindPeaksKwargs
from hplc_py.io_validation import IOValid

from ..map_peaks.map_peaks import MapPeaks


class MapWindows(IOValid):
    def __init__(
        self,
        prominence: float = 0.01,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ):
        self.X_key = "X"
        self.X_idx_key = "X_idx"

        self.w_type_interpeak_label: int = -999
        self.mp = MapPeaks(
            prominence=prominence, wlen=wlen, find_peaks_kwargs=find_peaks_kwargs
        )

        self.X_w = pl.DataFrame()  # type: ignore

    def fit(self, X: DataFrame[X_Schema], y=None) -> Self:

        self.X = X
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

        self.X_w: DataFrame[X_Windowed] = window_X(
            X=self.X,
            peak_wdw_intvls=peak_wdw_intvls,
            left_bases=left_bases,
            right_bases=right_bases,
            w_type_key=self.w_type_key,
            w_idx_key=self.w_idx_key,
        )

        return self


def _is_not_empty(
    ndframe: pd.DataFrame | pd.Series,
) -> None:
    if ndframe.empty:
        raise ValueError(f"{ndframe.name} is empty")


def _is_a_series(
    s: pd.Series,
) -> None:
    if not isinstance(s, pd.Series):
        if s.empty:
            raise ValueError(f"{s.name} is empty")


def _is_an_int_series(
    s: pd.Series,
) -> None:
    if not pd.api.types.is_integer_dtype(s):
        raise TypeError(f"Expected {s.name} to be int dtype, got {s.dtype}")


def _is_an_interval_series(
    s: pd.Series,
) -> None:
    if not isinstance(s.dtype, pd.IntervalDtype):
        raise TypeError(f"Expected {s.name} to be Interval dtype, got {s.dtype}")


def peak_base_intvl_factory(
    left_bases: Series[int64],
    right_bases: Series[int64],
) -> Series[pd.Interval]:
    """
    Take the left an right base index arrays and convert them to pandas Intervals, returned as a frame. Assumes that the left and right base arrays are aligned in order
    """
    # checks

    # check that s is a series
    # left and right base are int arrays

    # to convert the left and right bases to series, need to zip them together then use each pair as the closed bounds.

    intvls: pd.Series[pd.Interval] = pd.Series(name="peak_intervals")

    for i, pair in enumerate(zip(left_bases, right_bases)):
        intvls[i] = pd.Interval(pair[0], pair[1], closed="both")

    return intvls  # type: ignore


def map_wdws_to_peaks(
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
    _is_a_series(intvls)
    _is_an_interval_series(intvls)
    _is_not_empty(intvls)

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
        [Interval(intvl.left, intvl.right, i) for i, intvl in enumerate(intvls)]  # type: ignore
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


def set_wdw_intvls_from_peak_intvls(
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
            w_intvl = pd.Interval(intvls[wdws[0]].left, intvls[wdws[-1]].right, "both")
            w_intvls[w_idx] = w_intvl

        else:
            raise ValueError("expected a value for every window")

    return w_intvls


@pa.check_types
def peak_intvls_as_frame(
    peak_wdw_intvls: dict[int, pd.Interval],
    X_idx_key: str,
    w_type_key: str,
    w_idx_key: str,
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
        p_wdw = intvl_as_frame(
            i=i,
            intvl=intvl,
            X_idx_key=X_idx_key,
            w_type_key=w_type_key,
            w_idx_key=w_idx_key,
        )
        p_wdw_list.append(p_wdw)

    p_wdws_ = pd.concat(p_wdw_list).reset_index(drop=True).rename_axis(index="idx")

    p_wdws = DataFrame[PeakWindows](p_wdws_)

    return p_wdws


def intvl_as_frame(
    i,
    intvl,
    w_type_key: str,
    w_idx_key: str,
    X_idx_key: str,
):
    X_idx: NDArray[int64] = np.arange(intvl.left, intvl.right, 1)

    p_wdw: pd.DataFrame = pd.DataFrame(
        {
            X_idx_key: X_idx,
        }
    )

    p_wdw[w_type_key] = "peak"
    p_wdw[w_idx_key] = i

    p_wdw = p_wdw[[w_type_key, w_idx_key, X_idx_key]]

    p_wdw = p_wdw.astype(
        {
            w_idx_key: int64,
            w_type_key: pd.StringDtype(),
            w_idx_key: int64,
        }
    )
    return p_wdw


@pa.check_types
def set_peak_wndwd_X_idx(
    X: DataFrame[X_Schema],
    X_key: str,
    X_idx_pw: DataFrame[PeakWindows],
    X_idx_key: str,
    w_type_key: str,
    w_idx_key: str,
    null_fill: float = -999,
) -> DataFrame[X_PeakWindowed]:
    """
    Broadcasts the identified peak windows to the length of the series X idx axis.
    Any time without an assigned window is labeled as `w_type_interpeak_label`,
    i.e. -999 to indicate 'missing'. The patterns of missing values are later computed
    to label the interpeak windows.
    """
    if isinstance(X, DataFrame):
        X_pl: pl.DataFrame = pl.from_pandas(X)
    # joins the peak window intervals to the time index to generate a pattern of missing values

    p_wdws_: pl.DataFrame = pl.from_pandas(
        X_idx_pw, schema_overrides={"X_idx": pl.UInt32()}
    )

    # left join time idx to peak windows and peak types, leaving na's to be filled
    # with a placeholder and 'interpeak', respectively.

    X_pw_broadcast = (
        X_pl.with_row_index(X_idx_key)
        .join(p_wdws_, on=X_idx_key, how="left")
        .drop(X_idx_key)
    )
    # not here

    # this simply fills the nulls
    x_w_ = X_pw_broadcast.with_columns(
        **{
            w_type_key: pl.col(w_type_key).fill_null("interpeak"),
            w_idx_key: pl.col(w_idx_key).fill_null(null_fill),
        }
    ).select(w_type_key, w_idx_key, X_key)

    x_w_pd = x_w_.to_pandas()

    X_PeakWindowed.validate(x_w_pd, lazy=True)

    return DataFrame[X_PeakWindowed](x_w_pd)


def _get_interpeak_w_idxs(
    pwdwd_time: DataFrame[X_PeakWindowed],
    w_idx_key: str,
    null_fill: float,
) -> NDArray[int64]:
    labels = []
    labels, num_features = label(
        pwdwd_time[w_idx_key] == null_fill
    )  # type: ignore
    labels = np.asarray(labels, dtype=np.int64)
    return labels


# @pa.check_types
def label_interpeaks(
    X_pw: DataFrame[X_PeakWindowed],
    w_idx_key: str,
    null_fill: float,
) -> DataFrame[X_Windowed]:
    """
    Simply replaces the interpeak placeholder index with the mapping obtained from
    "get_na_labels"
    """

    X_w_: pd.DataFrame = X_pw.copy(deep=True)

    labels = _get_interpeak_w_idxs(
        X_pw, w_idx_key=w_idx_key, null_fill=null_fill
    )

    X_w_[w_idx_key] = X_w_[w_idx_key].mask(
        X_w_[w_idx_key] == null_fill,
        Series(labels - 1),
    )

    X_w_ = X_w_.rename_axis(index="idx")

    X_Windowed.validate(X_w_, lazy=True)

    X_w: DataFrame[X_Windowed] = DataFrame[X_Windowed](X_w_)

    return X_w


def window_X(
    X: DataFrame[X_Schema],
    left_bases: Series[float],
    right_bases: Series[float],
    X_key: str,
    X_idx_key: str,
    w_type_key: str = "w_type",
    w_idx_key: str = "w_idx",
    null_fill: float = -9999,
) -> DataFrame[X_Windowed]:
    """
    Window X by broadcasting the identified peak window intervals to the length
    of the signal. Patterns in missing window labels are coded as interpeak with
    their own ascending index.
    """
    peak_wdw_intvls = p_wdw_intvl_factory(left_bases, right_bases)

    # X_idw_w is the signal indices derived from the interval objects, rather than
    # directly from X.
    X_idx_pw: DataFrame[PeakWindows] = peak_intvls_as_frame(
        peak_wdw_intvls=peak_wdw_intvls,
        X_idx_key=X_idx_key,
        w_type_key=w_type_key,
        w_idx_key=w_idx_key,
        )
    # X_pw is the result of joining X with X_idx_w, with an arbitrary label where
    # interpeak windows are present
    X_pw: DataFrame[X_PeakWindowed] = set_peak_wndwd_X_idx(
        X=X,
        X_key=X_key,
        X_idx_pw=X_idx_pw,
        X_idx_key=X_idx_key,
        w_type_key=w_type_key,
        w_idx_key=w_idx_key,
        null_fill=null_fill,
    )

    X_w: DataFrame[X_Windowed] = label_interpeaks(
        X_pw=X_pw,
        w_idx_key=w_idx_key,
        null_fill=null_fill,
    )

    return X_w


def p_wdw_intvl_factory(left_bases, right_bases):
    left_bases_, right_bases_ = _check_bases(left_bases, right_bases)

    # get the peak bases as pd.Intveral objects, then map the peaks to windows
    # depending on whether they overlap
    # then construct the window interval objects.
    pb_intvls = peak_base_intvl_factory(left_bases_, right_bases_)
    wdw_peak_mapping: dict[int, list[int]] = map_wdws_to_peaks(pb_intvls)
    peak_wdw_intvls: dict[int, pd.Interval] = set_wdw_intvls_from_peak_intvls(
        pb_intvls, wdw_peak_mapping
    )
    return peak_wdw_intvls


def _check_bases(left_bases, right_bases):
    left_bases_: Series[int64] = pd.Series(left_bases, dtype=int64)  # type: ignore
    right_bases_: Series[int64] = pd.Series(right_bases, dtype=int64)  # type: ignore

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
