import duckdb as db
import numpy as np
import pandas as pd
import pandera as pa
import polars as pl
from matplotlib.axes import Axes as Axes
from numpy import int64
from numpy.typing import NDArray
from pandera.typing import Series
from pandera.typing.polars import DataFrame
from scipy.ndimage import label
from sklearn.base import BaseEstimator, TransformerMixin

from hplc_py.common.common_schemas import X_Schema
from hplc_py.map_signal.map_windows.window_map_output import WindowMap
from . import definitions as mw_defs
from .definitions import KEY_END, KEY_START, W_TYPE
from .schemas import (
    PeakWindows,
    WindowBounds,
    X_PeakWindowed,
    X_Windowed,
)

from .schemas import (
    InterpeakWindowStarts,
    PeakIntervalBounds,
    WindowedPeakIntervalBounds,
    WindowedPeakIntervals,
    WindowPeakMap,
)


class WindowMapper(TransformerMixin, BaseEstimator):

    @pa.check_types(lazy=True)
    def __init__(self):

        self.w_type_interpeak_label: int = -999

    def fit(self, X, y=None, left_bases=None, right_bases=None):

        self.X = X
        self.left_bases = left_bases
        self.right_bases = right_bases

        return self

    def transform(self, X, y=None):

        self.X = X

        self.window_map: WindowMap = window_X(
            X=X,
            left_bases=self.left_bases.to_pandas(),
            right_bases=self.right_bases.to_pandas(),
        )

        return self.X_windowed_

    @property
    def X_windowed_(self):
        return self.window_map.X_windowed

    @property
    def window_bounds_(self):
        return self.window_map.window_bounds

    def viz(self, opt_args={}):
        return self.vizzer.draw_peak_windows(
            bounds=self.window_bounds.filter(pl.col(W_TYPE).eq("peak")),
            opt_args=opt_args,
        )


def peak_base_intvl_factory(
    left_bases: Series[int],
    right_bases: Series[int],
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


@pa.check_types
def map_wdws_to_peaks(
    left_bases: Series[int],
    right_bases: Series[int],
) -> DataFrame[WindowedPeakIntervals]:
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

    intvls: Series[pd.Interval] = peak_base_intvl_factory(left_bases, right_bases)

    wdw_peak_mapping: dict[int, list[int]] = map_windows_to_peaks(intvls)

    intvl_frame: DataFrame[PeakIntervalBounds] = intervals_to_columns(intvls=intvls)

    window_peak_map: DataFrame[WindowPeakMap] = window_peak_map_as_frame(
        window_peak_mapping=wdw_peak_mapping
    )

    wdwd_peak_intervals: DataFrame[WindowedPeakIntervalBounds] = (
        join_intervals_to_window_peak_map(
            intvl_frame=intvl_frame, window_peak_map=window_peak_map
        )
    )

    return DataFrame[WindowedPeakIntervals](wdwd_peak_intervals)


def map_windows_to_peaks(intvls: Series[pd.Interval]) -> dict[int, list[int]]:
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
    wdw_peak_mapping: dict[int, list[int]] = {0: [Intvls[0].p_idx]}
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

    # sort the mappings

    for idx, peaks in wdw_peak_mapping.items():
        wdw_peak_mapping[idx] = sorted(peaks)

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
    return w_intvls  # type: ignore


@pa.check_types
def peak_intvls_as_frame(
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
        p_wdw: pd.DataFrame = pd.DataFrame(
            {
                mw_defs.W_TYPE: "peak",
                mw_defs.W_IDX: i,
                mw_defs.IDX: np.arange(intvl.left, intvl.right, 1),
            }
        ).astype(
            {
                mw_defs.W_IDX: int,
                mw_defs.W_TYPE: pd.StringDtype(),
                mw_defs.IDX: int,
            }
        )
        p_wdw_list.append(p_wdw)

    p_wdws_ = pd.concat(p_wdw_list).reset_index(drop=True)

    p_wdws = DataFrame[PeakWindows](p_wdws_)

    return p_wdws


@pa.check_types
def set_peak_wndwd_X_idx(
    X: DataFrame[X_Schema],
    X_idx_pw: DataFrame[PeakWindows],
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
        X_idx_pw, schema_overrides={mw_defs.IDX: pl.Int64()}
    )

    # left join time idx to peak windows and peak types, leaving na's to be filled
    # with a placeholder and 'interpeak', respectively.

    X_pw_broadcast = X_pl.join(p_wdws_, on=mw_defs.IDX, how="left")

    # this simply fills the nulls
    x_w_ = X_pw_broadcast.with_columns(
        **{
            mw_defs.W_TYPE: pl.col(mw_defs.W_TYPE).fill_null("interpeak"),
            mw_defs.W_IDX: pl.col(mw_defs.W_IDX).fill_null(null_fill),
        }
    ).select(mw_defs.W_TYPE, mw_defs.W_IDX, mw_defs.IDX, mw_defs.X)

    x_w_pd = x_w_.to_pandas()

    X_PeakWindowed.validate(x_w_pd, lazy=True)

    return DataFrame[X_PeakWindowed](x_w_pd)


def _get_interpeak_w_idxs(
    pwdwd_time: DataFrame[X_PeakWindowed],
    null_fill: float,
) -> NDArray[int64]:
    labels = []
    labels, num_features = label(pwdwd_time[mw_defs.W_IDX] == null_fill)  # type: ignore
    labels = np.asarray(labels, dtype=int)
    return labels


@pa.check_types
def label_interpeaks(
    X_pw: DataFrame[X_PeakWindowed],
    null_fill: float,
) -> DataFrame[X_Windowed]:
    """
    Simply replaces the interpeak placeholder index with the mapping obtained from
    "get_na_labels"
    """

    X_w_: pd.DataFrame = X_pw.copy(deep=True)

    labels = _get_interpeak_w_idxs(X_pw, null_fill=null_fill)

    X_w_[mw_defs.W_IDX] = X_w_[mw_defs.W_IDX].mask(
        X_w_[mw_defs.W_IDX] == null_fill,
        Series(labels - 1),
    )

    X_w_ = X_w_.rename_axis(index="idx")

    X_Windowed.validate(X_w_, lazy=True)

    X_w: DataFrame[X_Windowed] = DataFrame[X_Windowed](X_w_)

    return X_w


@pa.check_types
def window_X(
    X: DataFrame[X_Schema],
    left_bases: Series[float],
    right_bases: Series[float],
) -> WindowMap:
    """
    Window X by broadcasting the identified peak window intervals to the length
    of the signal. Patterns in missing window labels are coded as interpeak with
    their own ascending index.
    """
    assert isinstance(left_bases, pd.Series)
    assert isinstance(right_bases, pd.Series)
    # cast to int and round to match precision of index, i.e. 1.

    left_bases_int: Series[int]
    right_bases_int: Series[int]
    left_bases_int, right_bases_int = bases_as_int(
        left_bases=left_bases, right_bases=right_bases
    )

    wdwd_peak_intervals: DataFrame[WindowedPeakIntervals] = map_wdws_to_peaks(
        left_bases=left_bases_int,
        right_bases=right_bases_int,
    )

    X_peak_wdwd: DataFrame[X_PeakWindowed] = X.pipe(
        join_windowed_intervals_to_X,
        wdwd_peak_intervals=wdwd_peak_intervals,
    )

    X_w: DataFrame[X_Windowed] = label_interpeak_windows(
        X_peak_wdwd=X_peak_wdwd,
    )

    window_map: WindowMaap = WindowMap(X_windowed=X_w)

    return window_map


@pa.check_types
def bases_as_int(
    left_bases: Series[float],
    right_bases: Series[float],
) -> tuple[Series[int], Series[int]]:

    bases = pd.concat([left_bases, right_bases], axis=1)

    bases = bases.transform(round_half_away_from_zero)
    output = bases.pb_left, bases.pb_right

    return output


def round_half_away_from_zero(
    x,
    observe_intermeds: bool = False,
):
    """
    Take arrays of float and round using "rounding toward infinity" strategy to return
    arrays of int.

    # Notes

    python by default uses a rounding strategy called "rounding half to even", rounding up or down to even numbers to account for rounding errors, see [this](https://stackoverflow.com/questions/33019698/how-to-properly-round-up-half-float-numbers) and the [round docs](https://docs.python.org/3/library/functions.html#round)

    Synonyms for this strategy include:
    > convergent rounding, statistician's rounding, Dutch rounding, Gaussian rounding,
    > odd–even rounding or bankers' rounding.
    Source: [Rounding half to even](https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even)

    Generally speaking there is no ideal here, but in the sciences we expect 0.5 to round to 1, and -0.5 to round to -1, and so forth. This is called "rounding half away from zero", or "[rounding toward infinity](https://en.wikipedia.org/wiki/Rounding#Rounding_half_away_from_zero)".

    For now we'll implement a numpy strategy found [on stack overflow](https://stackoverflow.com/a/59142875/21058408).

    This method relies on doubling the fractional part of the float to see whether it rounds up or down. if the double is greater than 1, i.e. anything greater than or equal to 0.5, when double is floored, it is now equal to 1. Thus adding this value to the floor of x will shift it toward infinity if the fractional component meets the prior condition. Finally, to handle negatives, we first find the absolute value of x, and the signs of x as an array. The aformentioned calculation is performed on abs(x), then the result is multiplied by the sign array, preserving the direction of x.

    :math: tform = sign(x) * (floor((abs(x))+floor(2*(abs(x)%1)))
    """

    # first

    # modulo operator returns the remainder of a division. Dividing a float by 1
    # simply returns the fractional part of the number.
    if observe_intermeds:
        round_half_away_from_zero_observe_intermeds(x)
    else:
        tform = np.copysign(
            np.add(
                np.floor((np.abs(x))),
                np.floor(np.multiply(2, np.mod(np.abs(x), 1))),
            ),
            x,
        ).astype(int)
    return tform


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

    for idx, grp in window_peak_map_frame.group_by([mw_defs.W_IDX]):
        dict_peaks = window_peak_map_dict[idx[0]]  # type: ignore
        isin = grp.select(pl.col(mw_defs.P_IDX).is_in(dict_peaks).all()).item()
        assert isin


@pa.check_types
def intervals_to_columns(intvls: Series[pd.Interval]) -> DataFrame[PeakIntervalBounds]:
    """
    Take a series of Interval objects and return a frame with 'p_idx', 'left', and 'right'
    bound columns.
    """
    intvl_df = pl.DataFrame()

    intvl_idx = pd.IntervalIndex(intvls)  # type: ignore

    from IPython.core.debugger import set_trace

    intvl_df = (
        pl.DataFrame(
            {
                mw_defs.P_IDX: list(intvls.index),
                "left": list(intvl_idx.left),
                "right": list(intvl_idx.right),
            }
        )
        .pipe(PeakIntervalBounds.validate, lazy=True)
        .pipe(DataFrame[PeakIntervalBounds])
    )

    return intvl_df


def window_peak_map_as_frame(
    window_peak_mapping: dict[int, list[int]]
) -> DataFrame[WindowPeakMap]:
    """
    take a dict whose keys are window idxs and values are lists of assigned peak
    idxs and return a pandas frame of two columns, p_idx, and w_idx, where p_idx
    are unique and w_idx may repeat. This is to label each p_idx ready to join to
    the interval frame.
    """
    p_idx = sorted(p for peaks in window_peak_mapping.values() for p in peaks)
    window_peak_map = pl.DataFrame(
        {
            mw_defs.P_IDX: p_idx,
        }
    ).with_columns(pl.lit(0).cast(int).alias(mw_defs.W_IDX))

    # create a frame with p_idx and an empty w_idx. Iterate through the dict and
    # assign w_idx to the column at rows if the row p_idx is in the values of w_idx

    for w_idx_, peaks in window_peak_mapping.items():

        window_peak_map = window_peak_map.with_columns(
            pl.when(pl.col(mw_defs.P_IDX).is_in(peaks))
            .then(w_idx_)
            .otherwise(pl.col(mw_defs.W_IDX))
            .alias(mw_defs.W_IDX)
        )

    sanity_check_compare_frame_dict(
        window_peak_map_dict=window_peak_mapping, window_peak_map_frame=window_peak_map
    )

    return DataFrame[WindowPeakMap](window_peak_map.to_pandas())


def join_intervals_to_window_peak_map(
    intvl_frame: DataFrame[PeakIntervalBounds],
    window_peak_map: DataFrame[WindowPeakMap],
) -> DataFrame[WindowedPeakIntervalBounds]:

    joined = intvl_frame.join(
        window_peak_map, on=mw_defs.P_IDX, how="inner", validate="1:1"
    ).select(pl.col([mw_defs.W_IDX, mw_defs.P_IDX, "left", "right"]))

    return joined.pipe(DataFrame[WindowedPeakIntervalBounds])


@pa.check_types
def join_windowed_intervals_to_X(
    X: DataFrame[X_Schema],
    wdwd_peak_intervals: DataFrame[WindowedPeakIntervals],
) -> DataFrame[X_PeakWindowed]:
    """
    Given a left and right bound of each peak and window, asof join to the index such that
    the closest index to each interpolated idx value is matched, i.e. left and right,
    and then fill between.

    2024-02-14 15:57:52

    Modification - have removed ASOF join and rounded bases to int prior to entering
    this scope. Now we just need to arrange them in a suitable fashion for the pivot
    function and the rest should work as normal. Or skip the pivot because we dont need it,
    just need to
    """
    from polars import exceptions as pl_exceptions

    try:
        wdw_x_idx_bounds: pl.DataFrame = get_window_X_idx_bounds(
            wdwd_peak_intervals=wdwd_peak_intervals
        )
    except pl_exceptions.ColumnNotFoundError as e:
        e.add_note(f"\nwdwd_peak_intervals:\n {wdwd_peak_intervals.head()}")
        raise e

    X_peak_wdwd: DataFrame[X_PeakWindowed] = (
        map_peak_window_ranges_to_x_idx_fill_between(
            X=X,
            wdw_idx_bounds=wdw_x_idx_bounds,
        )
    )

    return DataFrame[X_PeakWindowed](X_peak_wdwd)


def get_window_X_idx_bounds(
    wdwd_peak_intervals: DataFrame[WindowedPeakIntervals],
) -> pl.DataFrame:
    """
    Given a frame of 'w_idx', 'p_idx', 'left', 'right', where the 'left' and 'right' are the peak interval bounds, return the bounds of each window.

    This is achieved through grouping by 'w_idx' and returning the first left and last right for each group

    TODO: type annotate, parametrize keys
    """

    wdw_X_idx_bounds = (
        wdwd_peak_intervals
        .group_by([mw_defs.W_IDX], maintain_order=True)
        .agg(
            pl.col("left")
            .min()
            .alias("left"),
            pl.col("right")
            .max()
            .alias("right")
        )
        .sort("left")
    )  # fmt: skip

    return wdw_X_idx_bounds


def map_peak_window_ranges_to_x_idx_fill_between(
    X: DataFrame[X_Schema],
    wdw_idx_bounds: DataFrame[WindowBounds],
) -> DataFrame[X_PeakWindowed]:
    """
    maps the window index and peak type to the time range of each peak.

    Returns a long frame of three columns: 'X_idx', 'w_idx', and 'w_type' where
    'X_idx' is discontinuous
    """
    wdw_idx_bounds  # type: ignore

    query: str = f"""--sql
                   SELECT x.{mw_defs.IDX}, bds.{mw_defs.W_IDX}, x.{mw_defs.X}
                            FROM
                                X x
                            LEFT JOIN
                                wdw_idx_bounds bds
                            ON
                              x.{mw_defs.IDX}>=bds.left
                            AND
                             x.{mw_defs.IDX}<=bds.right
                            --WHERE
                              --bds.{mw_defs.W_IDX} IS DISTINCT FROM NULL
                            ORDER BY
                              {mw_defs.IDX}
                   """

    X_peak_windows: pl.DataFrame = (
          db.sql(query)
            .pl()
            .with_columns(
                pl.when(
                    pl.col(mw_defs.W_IDX)
                      .is_null())
                .then(
                    pl.lit("interpeak")
                    )
                .otherwise(
                    pl.lit("peak")
                    )
                .alias(mw_defs.W_TYPE)
            ).select(
                pl.col(mw_defs.W_TYPE, mw_defs.W_IDX, mw_defs.IDX, mw_defs.X)
                )
            )  # fmt: skip

    X_peak_windows.pipe(X_PeakWindowed.validate, lazy=True).pipe(
        DataFrame[X_PeakWindowed]
    )

    return X_peak_windows
    # return peak_windows


@pa.check_types
def label_interpeak_windows(
    X_peak_wdwd: DataFrame[X_PeakWindowed],
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

    # get the start of each interpeak window and label by ordinal rank of X_idx
    interpeak_wdw_starts: DataFrame[InterpeakWindowStarts] = (
        find_interpeak_window_starts(
            X_peak_wdwd=X_peak_wdwd,
        )
    )

    # join back to X_peak_windowed
    X_windowed = label_interpeak_window_idx(
        X_peak_wdwd=X_peak_wdwd,
        interpeak_wdw_starts=interpeak_wdw_starts,
    )

    return DataFrame[X_Windowed](X_windowed)


@pa.check_types(lazy=True)
def label_interpeak_window_idx(
    X_peak_wdwd: DataFrame[X_PeakWindowed],
    interpeak_wdw_starts: DataFrame[InterpeakWindowStarts],
) -> DataFrame[X_Windowed]:
    """
    Join the labeled interpeak window starts to p_wdws then foreward fill the missing
    values to label the interpeak ranges.
    """
    X_windowed = (
        X_peak_wdwd.pipe(
            left_join_p_wdws_interpeak_wdw_starts,
            interpeak_wdw_starts=interpeak_wdw_starts,
        )
        .with_columns(
            pl.col(mw_defs.W_IDX).replace({999999: None}).alias(mw_defs.W_IDX)
        )
        .with_columns(
            pl.col(mw_defs.W_IDX)
            .fill_null(pl.col(f"{mw_defs.W_IDX}_right"))
            .alias(mw_defs.W_IDX)
        )
        .with_columns(pl.col(mw_defs.W_IDX).forward_fill().alias(mw_defs.W_IDX))
        .drop(f"{mw_defs.W_IDX}_right")
        .pipe(X_Windowed.validate, lazy=True)
        .pipe(DataFrame[X_Windowed])
    )

    return X_windowed


@pa.check_types
def left_join_p_wdws_interpeak_wdw_starts(
    X_p_wdwd: pl.DataFrame,
    interpeak_wdw_starts: pl.DataFrame,
) -> pl.DataFrame:
    """
    `X_p_wdwd` is the polars form of `X_PeakWindowed`, `interpeak_wdw_starts` is the polars form of `InterpeakWindowStarts`.

    Note: Cannot type annotate this function with Pandera as it is used in a Polars pipeline.
    """

    interpeak_wdw_starts_: pl.DataFrame = interpeak_wdw_starts.select(
        pl.exclude(["X_idx_diff", mw_defs.X])
    )

    out = X_p_wdwd.join(
        interpeak_wdw_starts_,
        on=[mw_defs.W_TYPE, mw_defs.IDX],
        how="left",
    )

    return out


@pa.check_types
def find_interpeak_window_starts(
    X_peak_wdwd: DataFrame[X_PeakWindowed],
) -> DataFrame[InterpeakWindowStarts]:
    """
    Find the start idx of each interpeak window by first finding indices where
    the discrete difference between n and n+1 is not equal to 1 (a condition which allows
    us to include the first index). This allows us to filter all values which are not
    start indices, then ordinal rank by time idx to label each start index with an
    ascendingwindow idx.
    """

    # note: fill_null = 0 is arbirary, simply need to be be a value that is caught by
    # the !=1 condition.

    def check_for_boundary_condition(X_peak_wdwd: DataFrame[X_PeakWindowed]):
        """
        As of 2024-04-17 09:28:12, this module treats the x axis as a rational number plane.
        The problem with this, if we want unique idx values in the table, is that we cannot
        include windows of length = 1, as they will start where the previous window ends,
        and etc. I consider this a boundary issue, as the resolution of the x axis
        is sufficiently high, and is primarily occuring due the artifically high precison
        of the floating point input. This function will raise a user error if an interpeak
        window of length 1 is possible, just in case it pops up again.
        """

        # find the peak window bounds, left and right

        peak_wdw_bounds = (
            X_peak_wdwd.filter(pl.col("w_type").eq("peak"))
            .groupby("w_idx", maintain_order=True)
            .agg(
                pl.col("idx").first().alias("left"), pl.col("idx").last().alias("right")
            )
        )

        # find peak windows who are next to each other on the real number plane, defined
        # as having a time index difference of 1.

        find_length_1_tbl = (
            peak_wdw_bounds.with_columns(
                pl.col("left").shift(-1),
                pl.col("w_idx").shift(-1).alias("w_idx_left"),
                pl.col("w_idx").alias("w_idx_right"),
            )
            .with_columns(pl.col("left").sub(pl.col("right")).alias("diff"))
            .with_columns(
                pl.when(pl.col("diff").eq(1))
                .then(True)
                .otherwise(False)
                .alias("is_length_1")
            )
        )

        # filter to 'is_length_1'

        length_1_rows = find_length_1_tbl.filter(pl.col("is_length_1").eq(True))

        # note: left and right are swapped because this is from the frame of reference of the
        # interpeak windows rather than the source windows.

        warning_output_tbl = length_1_rows.select(
            pl.col("right").alias("left"),
            pl.col("left").alias("right"),
            pl.col("w_idx_right").alias("w_idx_left"),
            pl.col("w_idx_left").alias("w_idx_right"),
        )

        if not warning_output_tbl.is_empty():
            warning_str = (
                f"interpeak window of length one detected:\n{warning_output_tbl}\n"
            )
            warning_str += "left: the interpeak window start idx, right: interpeak window end idx, w_idx_left: the peak window idx on the left, w_idx_right: the peak window idx on the right.\n"

            warning_str += "\nthis is a development warning as I do not expect this scenario to occur if a sufficiently low level of precision is used. It is happening (i think) because the peak measuring algorithms are very sensitive to precision.\n"

            warning_str += "\nDo with it what you will, but the result is neighbouring peak windows with no intermediate interpeak window\n"

            import warnings

            warnings.warn(warning_str)

    check_for_boundary_condition(X_peak_wdwd=X_peak_wdwd)

    interpeak_wdw_starts = (
        X_peak_wdwd
        .filter(pl.col(mw_defs.W_TYPE) == mw_defs.LABEL_INTERPEAK)
        .with_columns(
            pl.col(mw_defs.IDX)
            .diff()
            .fill_null(0)
            .alias(f"{mw_defs.IDX}_diff"))
        .filter(pl.col(f"{mw_defs.IDX}_diff") != 1)
        .with_columns(
            pl.col(mw_defs.IDX)
            .rank("ordinal")
            .sub(1)
            .cast(int)
            .alias(mw_defs.W_IDX)
        )
        .drop([f"{mw_defs.IDX}_diff"])
    )  # fmt: skip

    return interpeak_wdw_starts.pipe(DataFrame[InterpeakWindowStarts])


def round_half_away_from_zero_observe_intermeds(x) -> None:
    """
    perform the rounding algorithm through polars, storing the intermediate values
    for observation.
    """
    x_pl = pl.from_pandas(x).to_frame()
    x_pl = (
        x_pl.with_columns(
            pl.col("x").abs().alias("a"),
        )
        .with_columns(
            pl.col("a").mod(1).alias("fract"),
        )
        .with_columns(pl.col("fract").mul(2).alias("dbl_fract"))
        .with_columns(pl.col("a").floor().alias("floor_a"))
        .with_columns(pl.col("dbl_fract").floor().alias("floor_dbl_fract"))
        .with_columns(
            pl.sum_horizontal([pl.col("floor_a"), pl.col("floor_dbl_fract")]).alias(
                "floor_a+floor_dbl_fract"
            )
        )
        .with_columns(
            pl.col("x")
            .sign()
            .mul(pl.col("floor_a+floor_dbl_fract"))
            .alias("out")
            .cast(int)
        )
    )
