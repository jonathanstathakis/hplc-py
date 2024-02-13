import duckdb as db
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
    X_PeakWindowed,
    X_Schema,
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
        self.w_type_key = "w_type"
        self.w_idx_key = "w_idx"

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
            X_key=self.X_key,
            X_idx_key=self.X_idx_key,
            left_bases=left_bases,
            right_bases=right_bases,
            w_type_key=self.w_type_key,
            w_idx_key=self.w_idx_key,
        )

        return self


def peak_base_intvl_factory(
    left_bases: Series[float],
    right_bases: Series[float],
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
    left_bases: Series[float],
    right_bases: Series[float],
) -> pl.DataFrame:
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

    # group first then form the intervals second. use 'index' for the group values

    intvls = peak_base_intvl_factory(left_bases, right_bases)
    wdw_peak_mapping = map_windows_to_peaks(intvls)
    intvl_frame = intervals_to_columns(intvls=intvls)
    window_peak_map = window_peak_map_as_frame(window_peak_mapping=wdw_peak_mapping)
    windowed_intervals = join_intervals_to_window_peak_map(
        intvl_frame=intvl_frame, window_peak_map=window_peak_map
    )

    return windowed_intervals


def map_windows_to_peaks(intvls):
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

    # checks
    _is_a_series(intvls)
    _is_an_interval_series(intvls)
    _is_not_empty(intvls)

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

    intvl_frame = intervals_to_columns(intvls)
    window_peak_map = window_peak_map_as_frame(w_idxs)
    windowed_intervals = join_intervals_to_window_peak_map(
        intvl_frame=intvl_frame, window_peak_map=window_peak_map
    )

    for w_idx, peaks in w_idxs.items():
        if len(peaks) == 1:
            w_intvls[w_idx] = intvls[peaks[0]]
        elif len(peaks) > 1:
            w_intvls = intvls[min(peaks) : max(peaks)]

            wdw_left = intvls[peaks[0]].left
            wdw_right = intvls[peaks[-1]].right
            w_intvl = pd.Interval(wdw_left, wdw_right, "both")
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
        p_wdw: pd.DataFrame = pd.DataFrame(
            {
                w_type_key: "peak",
                w_idx_key: i,
                X_idx_key: np.arange(intvl.left, intvl.right, 1),
            }
        ).astype(
            {
                w_idx_key: int64,
                w_type_key: pd.StringDtype(),
                w_idx_key: int64,
            }
        )
        p_wdw_list.append(p_wdw)

    p_wdws_ = pd.concat(p_wdw_list).reset_index(drop=True).rename_axis(index="idx")

    p_wdws = DataFrame[PeakWindows](p_wdws_)

    return p_wdws


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
        X_idx_pw, schema_overrides={"X_idx": pl.Int64()}
    )

    # left join time idx to peak windows and peak types, leaving na's to be filled
    # with a placeholder and 'interpeak', respectively.

    X_pw_broadcast = X_pl.join(p_wdws_, on=X_idx_key, how="left")

    # this simply fills the nulls
    x_w_ = X_pw_broadcast.with_columns(
        **{
            w_type_key: pl.col(w_type_key).fill_null("interpeak"),
            w_idx_key: pl.col(w_idx_key).fill_null(null_fill),
        }
    ).select(w_type_key, w_idx_key, X_idx_key, X_key)

    x_w_pd = x_w_.to_pandas()

    X_PeakWindowed.validate(x_w_pd, lazy=True)

    return DataFrame[X_PeakWindowed](x_w_pd)


def _get_interpeak_w_idxs(
    pwdwd_time: DataFrame[X_PeakWindowed],
    w_idx_key: str,
    null_fill: float,
) -> NDArray[int64]:
    labels = []
    labels, num_features = label(pwdwd_time[w_idx_key] == null_fill)  # type: ignore
    labels = np.asarray(labels, dtype=np.int64)
    return labels


@pa.check_types
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

    labels = _get_interpeak_w_idxs(X_pw, w_idx_key=w_idx_key, null_fill=null_fill)

    X_w_[w_idx_key] = X_w_[w_idx_key].mask(
        X_w_[w_idx_key] == null_fill,
        Series(labels - 1),
    )

    X_w_ = X_w_.rename_axis(index="idx")

    X_Windowed.validate(X_w_, lazy=True)

    X_w: DataFrame[X_Windowed] = DataFrame[X_Windowed](X_w_)
    breakpoint()
    return X_w


def window_X(
    X: DataFrame[X_Schema],
    left_bases: Series[float],
    right_bases: Series[float],
    X_key: str,
    X_idx_key: str,
    w_type_key: str = "w_type",
    w_idx_key: str = "w_idx",
) -> DataFrame[X_Windowed]:
    """
    Window X by broadcasting the identified peak window intervals to the length
    of the signal. Patterns in missing window labels are coded as interpeak with
    their own ascending index.
    """
    windowed_intervals: pl.DataFrame = map_wdws_to_peaks(
        left_bases=left_bases, right_bases=right_bases
    )

    X_peak_wdwd: DataFrame[X_PeakWindowed] = join_windowed_intervals_to_X(
        X=X,
        windowed_intervals=windowed_intervals,
        X_key=X_key,
        X_idx_key=X_idx_key,
        w_idx_key=w_idx_key,
        w_type_key=w_type_key,
        )

    X_w: DataFrame[X_Windowed] = label_interpeak_windows(
        X_peak_windowed=X_peak_wdwd,
        X_idx_key=X_idx_key,
        w_idx_key=w_idx_key,
        w_type_key=w_type_key,
    )

    return DataFrame[X_Windowed](X_w)


def p_wdw_intvl_factory(left_bases, right_bases):
    left_bases_, right_bases_ = _check_bases(left_bases, right_bases)

    # get the peak bases as pd.Intveral objects, then map the peaks to windows
    # depending on whether they overlap then construct the window interval objects.

    breakpoint()
    return wdw_peak_mapping


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


def sanity_check_compare_frame_dict(
    window_peak_map_dict: dict[int, list[int]],
    window_peak_map_frame: pl.DataFrame,
) -> None:
    """
    A sanity check to ensure the translation from dict to polars frame occured
    correctly. Iterate through the frame groupby on window idx and check if
    the p_idx values are equal.
    """
    idx: int
    grp: pl.DataFrame
    for idx, grp in window_peak_map_frame.group_by("w_idx"):

        dict_peaks = window_peak_map_dict[idx]
        isin = grp.select(pl.col("p_idx").is_in(dict_peaks).all()).item()
        assert isin


def intervals_to_columns(intvls: Series[pd.Interval]) -> pl.DataFrame:
    """
    Take a series of Interval objects and return a frame with 'p_idx', 'left', and 'right'
    bound columns.
    """
    intvl_df = pl.DataFrame()

    intvl_idx = pd.IntervalIndex(intvls)

    intvl_df = pl.DataFrame(
        {
            "p_idx": list(intvls.index),
            "left": list(intvl_idx.left),
            "right": list(intvl_idx.right),
        }
    )

    return intvl_df


def window_peak_map_as_frame(window_peak_mapping: dict[int, list[int]]) -> pl.DataFrame:
    """
    take a dict whose keys are window idxs and values are lists of assigned peak
    idxs and return a pandas frame of two columns, p_idx, and w_idx, where p_idx
    are unique and w_idx may repeat. This is to label each p_idx ready to join to
    the interval frame.
    """
    p_idx = sorted(p for peaks in window_peak_mapping.values() for p in peaks)
    window_peak_map = pl.DataFrame(
        {
            "p_idx": p_idx,
        }
    ).with_columns(w_idx=pl.lit(0))

    # create a frame with p_idx and an empty w_idx. Iterate through the dict and
    # assign w_idx to the column at rows if the row p_idx is in the values of w_idx

    for w_idx_, peaks in window_peak_mapping.items():

        window_peak_map = window_peak_map.with_columns(
            pl.when(pl.col("p_idx").is_in(peaks))
            .then(w_idx_)
            .otherwise(pl.col("w_idx"))
            .alias("w_idx")
        )

    sanity_check_compare_frame_dict(
        window_peak_map_dict=window_peak_mapping, window_peak_map_frame=window_peak_map
    )

    return window_peak_map


def join_intervals_to_window_peak_map(intvl_frame, window_peak_map):
    joined = pl.DataFrame()

    joined = intvl_frame.join(
        window_peak_map, on="p_idx", how="inner", validate="1:1"
    ).select(pl.col(["w_idx", "p_idx", "left", "right"]))

    return joined


@pa.check_types
def join_windowed_intervals_to_X(
    X: DataFrame[X_Schema],
    windowed_intervals: pl.DataFrame,
    X_key: str,
    X_idx_key: str,
    w_idx_key: str,
    w_type_key: str,
) -> DataFrame[X_PeakWindowed]:
    """
    Given a left and right bound of each peak and window, asof join to the index such that
    the closest index to each interpolated idx value is matched, i.e. left and right,
    and then fill between.
    """

    peak_wdws_x_idx = asof_join_wdws_to_time(X=X, windowed_intervals=windowed_intervals, X_idx_key=X_idx_key)

    wdw_idx_bounds = pivot_wdw_bounds(
        df=peak_wdws_x_idx, window_key=w_idx_key, time_key=X_idx_key
    )
    # join the window bounds to X, mapping the intervals betwen the start and finish of
    # each window with the w_idx.
    X_peak_wdwd = map_windows_to_peaks_on_signal_idx(
        X=X, window_idx_bounds=wdw_idx_bounds, X_key=X_key, X_idx_key=X_idx_key, w_idx_key=w_idx_key, w_type_key=w_type_key
    )

    return DataFrame[X_PeakWindowed](X_peak_wdwd)


@pa.check_types
def map_windows_to_peaks_on_signal_idx(
    X: DataFrame[X_Schema],
    window_idx_bounds: pl.DataFrame | pd.DataFrame,
    X_key: str,
    X_idx_key: str,
    w_idx_key: str,
    w_type_key: str,
) -> DataFrame[X_PeakWindowed]:
    """
    Use SQL range joins to simplify the mapping of peak windows to the signal index
    """

    # returns a three column table of 'X_idx','w_idx','w_type' of peak windows
    # vertically stacked in time order, and one value for 'w_type' = 'peak'

    peak_windows = (
        db.sql(
            f"""--sql
                   SELECT {X_idx_key}, {w_idx_key}
                            FROM
                                X x
                            LEFT OUTER JOIN
                                window_idx_bounds bds
                            ON
                                x.{X_idx_key}>=bds.start
                            AND
                                x.{X_idx_key}<=bds.end
                            WHERE
                                bds.{w_idx_key} IS DISTINCT FROM NULL
                            ORDER BY
                                {X_idx_key}
                   """
        )
        .pl()
        .with_columns(pl.lit("peak").alias(w_type_key))
    )

    # joins peak_windows to X producing NULL in 'w_idx' and 'w_type' where interpeak
    # regions lie.

    df_sql = db.sql(
        f"""--sql
                    SELECT
                    l.{X_idx_key}, l.{X_key}, r.{w_idx_key}, r.{w_type_key} 
                    FROM 
                        X l
                    LEFT JOIN                    
                        peak_windows r
                    ON
                        l.{X_idx_key}=r.{X_idx_key}
                    ORDER BY
                        l.{X_idx_key}
                    """
    )

    # label all null in w_type with 'interpeak'. Have to explicitely load the table
    # into the SQL server to do updates

    db.sql(
        f"""--sql
            CREATE TEMPORARY TABLE
            temp
            AS SELECT
            *
            FROM
            df_sql;
            
            UPDATE
                temp
            SET
                {w_type_key}='interpeak'
            WHERE
                {w_type_key}
            IS NOT DISTINCT
            FROM
                NULL
            """
    )

    # label all NULL w_idx with 999999 to make them distinct but not NULL.

    X_peak_windowed = db.sql(
        f"""--sql
           UPDATE
            temp
        SET
        {w_idx_key}=999999
            WHERE {w_idx_key}
            IS NOT DISTINCT
            FROM
                NULL;
        SELECT {w_type_key}, {w_idx_key}, {X_idx_key}, {X_key} FROM temp
                """
    ).df()

    X_PeakWindowed.validate(X_peak_windowed, lazy=True)

    return DataFrame[X_PeakWindowed](X_peak_windowed)


def pivot_wdw_bounds(df: pl.DataFrame, window_key: str, time_key: str):
    """
    For a long dataframe containing windows labelled by `window_key` in ranges,
    pivot the frame to produce a report of the start and stop `time_key` of each window,
    sorted by start time.
    """
    window_time_bounds = (
        df.group_by(window_key)
        .agg(start=pl.col(time_key).first(), end=pl.col(time_key).last())
        .sort("start")
    )
    return window_time_bounds


@pa.check_types
def asof_join_wdws_to_time(
    X: DataFrame[X_Schema],
    windowed_intervals: pl.DataFrame,
    X_idx_key: str,
) -> pl.DataFrame:
    """
    As the left and right ips are interpolated values derived from the integer space,
    we need to find the closest match for each. One method is via an ASOF join.
    the result is the bounds of each peak window labelled by w_idx in index units rather
    than interpolated units.

    I use Polars because it provides a 'nearest' strategy

    each input to asof is required to be sorted by the join key olars also requires
    explicit flagging of the sort column
    """
    X_pl = pl.from_pandas(X)

    asof_X = (
        X_pl.select(pl.col(X_idx_key).cast(pl.Float64).alias(X_idx_key))
        .sort(X_idx_key)
        .set_sorted(X_idx_key)
    )

    # do the same for the left and right columns
    asof_WI_left = (
        windowed_intervals.select(pl.exclude("right")).sort("left").set_sorted("left")
    )
    asof_WI_right = (
        windowed_intervals.select(pl.exclude("left")).sort("right").set_sorted("right")
    )

    X_peak_wdws_left = asof_WI_left.join_asof(
        asof_X,
        left_on="left",
        right_on=X_idx_key,
        strategy="nearest",
        tolerance=1,
    )

    X_peak_wdws_right = asof_WI_right.join_asof(
        asof_X,
        left_on="right",
        right_on=X_idx_key,
        strategy="nearest",
        tolerance=1,
    )

    p_wdws_x_idx = pl.concat(
        [
            X_peak_wdws_left.drop("left").with_columns(side=pl.lit("left")),
            X_peak_wdws_right.drop("right").with_columns(side=pl.lit("right")),
        ]
    ).with_columns(
        w_type=pl.lit("peak"),
        X_idx=pl.col(X_idx_key).cast(pl.Int64),
    )

    return p_wdws_x_idx


@pa.check_types
def label_interpeak_windows(
    X_peak_windowed: DataFrame[X_PeakWindowed],
    X_idx_key: str,
    w_idx_key: str,
    w_type_key: str,
) -> DataFrame[X_Windowed]:
    """
    index each NULL ranked by time order

    To do so I will need to:
    - select the NULL ranges
    - identify where each starts by finding jumps in X_idx
    - labeling the jumps
    - foreward fill
    - join back on main
    - merge the w_idx cols via WHERE
    """

    p_wdws = pl.from_pandas(X_peak_windowed)

    # get the start of each interpeak window and label by ordinal rank of X_idx
    interpeak_wdw_starts = find_interpeak_window_starts(
        p_wdws=p_wdws, w_type_key=w_type_key, w_idx_key=w_idx_key, X_idx_key=X_idx_key
    )

    # join back to X_peak_windowed
    X_windowed = label_interpeak_window_idx(
        p_wdws=p_wdws,
        interpeak_wdw_starts=interpeak_wdw_starts,
        w_idx_key=w_idx_key,
        X_idx_key=X_idx_key,
    )
    return DataFrame[X_Windowed](X_windowed)


def label_interpeak_window_idx(
    p_wdws: pl.DataFrame,
    interpeak_wdw_starts: pl.DataFrame,
    w_idx_key: str,
    X_idx_key: str,
) -> DataFrame[X_Windowed]:
    """
    Join the labeled interpeak window starts to p_wdws then foreward fill the missing
    values to label the interpeak ranges.
    """

    X_windowed = (
        p_wdws.join(
            interpeak_wdw_starts.select([w_idx_key, X_idx_key]),
            on=X_idx_key,
            how="left",
        )
        .with_columns(pl.col(w_idx_key).replace({999999: None}).alias(w_idx_key))
        .with_columns(
            pl.col(w_idx_key).fill_null(pl.col(f"{w_idx_key}_right")).alias(w_idx_key)
        )
        .with_columns(pl.col(w_idx_key).forward_fill().alias(w_idx_key))
        .drop(f"{w_idx_key}_right")
    )

    X_windowed_pd = X_windowed.to_pandas()

    X_Windowed.validate(X_windowed_pd, lazy=True)
    return DataFrame[X_Windowed](X_windowed_pd)


def find_interpeak_window_starts(
    p_wdws: pl.DataFrame, w_type_key: str, X_idx_key: str, w_idx_key: str
) -> pl.DataFrame:
    """
    Find the start idx of each interpeak window by first finding indices where
    the discrete difference between n and n+1 is not equal to 1 (a condition which allows
    us to include the first index). This allows us to filter all values which are not
    start indices, then ordinal rank by time idx to label each start index with an
    ascendingwindow idx.
    """

    # note: fill_null = 0 is arbirary, simply need to be be a value that is caught by
    # the !=1 condition.

    interpeak_wdw_starts = (
        p_wdws.filter(pl.col(w_type_key) == "interpeak")
        .with_columns(pl.col(X_idx_key).diff().fill_null(0).alias(f"{X_idx_key}_diff"))
        .filter(pl.col(f"{X_idx_key}_diff") != 1)
        .with_columns(pl.col(w_idx_key).rank("ordinal").sub(1).alias(w_idx_key))
    )

    return interpeak_wdw_starts


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
