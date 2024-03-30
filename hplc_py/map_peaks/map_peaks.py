"""
Contains methods relevant to mapping the peaks present in a signal, relying on `scipy.signal` for calculations.

NOTE: 2024-02-23 15:53:24 the `find_peaks` method of peak detection and base detection is VERY sensitive to the baseline smoothness. For example in the assessment_chrom dataset, even though the baseline correction produces no noticable adjustment in the signal, forgoing that stage results in the peak bases to be almost the length of the signal, on several of the peaks. This is presumably due to curvature and roughness in the baseline permitting the peak window lines to extend much further than they otherwise should. As SNIP smooths and corrects at the same time, it provides both functions. In conclusion, always correct and smooth, as if the peaks/peak regions do not return to zero (the implicit baseline) the windows will be much wider than desired and makes deconvolution impractical.
"""

from matplotlib.pyplot import contour
from numpy import float64, int64
from typeguard import typechecked
from numpy.typing import NDArray
from typing import Literal, Optional, TypeAlias, Self

import pandas as pd
import pandera as pa
from matplotlib.axes import Axes as Axes
from pandera.typing import DataFrame
from scipy import signal
from hplc_py.map_peaks.definitions import map_peaks_kwargs_defaults
from hplc_py.common.common_schemas import X_Schema

from hplc_py.map_peaks.definitions import MapPeaksKwargs
from hplc_py.map_peaks import definitions as mp_defs
from hplc_py.map_peaks import schemas as mp_schs
from hplc_py.map_peaks import viz as mp_viz

from hplc_py.map_peaks.schemas import (
    ContourLineBounds,
    Maxima,
    PeakMap,
    PeakMapOutput,
)

from typing import Tuple

from hplc_py.io_validation import IOValid
from hplc_py.map_peaks.schemas import WHH, FindPeaks, PeakBases

PPD: TypeAlias = Tuple[NDArray[float64], NDArray[int64], NDArray[int64]]
import polars as pl


class MapPeaks:
    """
    For a 1D array treatable as a signal, map peak properties returned as normalised dataframe tables and provide plotting methods.

    Notes:

    - Peak bases are measured twice - once in `find_peaks`, returned in the peak prominence data, and again with `find_widths` using a rel height of 1. For default settings, the bases as measured by `prominence` are wider than the latter, and do not correspond to the apparent profile of the peaks.
    """

    def __init__(
        self,
        X: DataFrame[X_Schema],
        find_peaks_kwargs: MapPeaksKwargs = map_peaks_kwargs_defaults,
    ):
        peak_mapper = PeakMapper(find_peaks_kwargs=find_peaks_kwargs)
        peak_mapper.fit(X=X)
        peak_mapper.transform()

        self.peak_map: PeakMapOutput = peak_mapper.peak_map

        self.plot = mp_viz.VizPeakMapFactory(X=X, peak_map=self.peak_map)


class PeakMapper(IOValid):

    def __init__(
        self,
        find_peaks_kwargs: MapPeaksKwargs = map_peaks_kwargs_defaults,
    ):
        """
        Use to map peaks, i.e. locate maxima, whh, peak bases for a given set of user inputs. Includes plotting functionality inherited from MapPeakPlots
        """
        self._find_peaks_kwargs = find_peaks_kwargs

    @pa.check_types()
    def fit(
        self,
        X: DataFrame[X_Schema],
        y=None,
    ) -> Self:
        """
        In this case, simply stores X and the timestep
        """

        self._X = X

        return self

    @pa.check_types()
    def transform(
        self,
    ) -> Self:

        self.peak_map = map_peaks(X=self._X, find_peaks_kwargs=self._find_peaks_kwargs)

        return self


def map_peaks(
    X: DataFrame[X_Schema],
    find_peaks_kwargs: mp_defs.MapPeaksKwargs = map_peaks_kwargs_defaults,
):
    """
    Map An input signal with peaks, providing peak height, prominence, and width data.

    From scipy.peak_widths docs:
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html>
    "wlen : int, optional A window length in samples passed to peak_prominences as an optional argument for internal calculation of prominence_data. This argument is ignored if prominence_data is given." - this is only used in the initial findpeaks
    then the results are propagated to the other profilings.
    """
    fp = set_findpeaks(
        X=X,
        find_peaks_kwargs=find_peaks_kwargs,
    )

    peak_prom_data = cast_peak_prom_data_to_tuple_of_numpy(fp=fp)

    peak_t_idx = fp[mp_defs.X_IDX].to_numpy(int)

    whh = width_df_factory(
        X=X,
        peak_t_idx=peak_t_idx,
        peak_prom_data=peak_prom_data,
        rel_height=0.5,
        key_X=mp_defs.X,
        key_width=mp_defs.KEY_WIDTH_WHH,
        key_left_ips=mp_defs.KEY_LEFT_WHH,
        key_right_ips=mp_defs.KEY_RIGHT_WHH,
        key_height=mp_defs.KEY_HEIGHT_WHH,
    )

    pb = width_df_factory(
        X=X,
        key_X=mp_defs.X,
        peak_t_idx=peak_t_idx,
        peak_prom_data=peak_prom_data,
        rel_height=1,
        key_width=mp_defs.KEY_WIDTH_PB,
        key_left_ips=mp_defs.KEY_LEFT_PB,
        key_right_ips=mp_defs.KEY_RIGHT_PB,
        key_height=mp_defs.KEY_HEIGHT_PB,
    )

    peak_map: PeakMapOutput = set_peak_map(
        X=X,
        fp=fp,
        whh=whh,
        pb=pb,
    )

    return peak_map


@pa.check_types
def set_findpeaks(
    X: DataFrame[X_Schema],
    find_peaks_kwargs: MapPeaksKwargs = MapPeaksKwargs(),
) -> DataFrame[FindPeaks]:

    if not isinstance(find_peaks_kwargs, dict):
        raise TypeError("Expect find_peaks_kwargs to be a dict")

    # move prominence to the X scale to avoid normalizing the signal
    find_peaks_kwargs[mp_defs.KEY_PROMINENCE] = (
        dict(find_peaks_kwargs).pop(mp_defs.KEY_PROMINENCE) * X[mp_defs.X].max()
    )

    X_idx, _dict = signal.find_peaks(
        x=X[mp_defs.X],
        **find_peaks_kwargs,
    )
    fp = (
        X.loc[X_idx]
        .assign(**{mp_defs.X_IDX: X_idx, **_dict})
        .reset_index(drop=True)
        .reset_index(names=mp_defs.P_IDX)
        .reset_index(drop=True)  # set each row number as the p_idx
        .rename(
            {
                "prominences": mp_defs.KEY_PROMINENCE,
                "left_bases": mp_defs.KEY_LEFT_PROM,
                "right_bases": mp_defs.KEY_RIGHT_PROM,
                mp_defs.X: mp_defs.MAXIMA,
            },
            axis=1,
        )
        .loc[
            :,
            lambda df: df.columns.drop(mp_defs.MAXIMA).insert(2, mp_defs.MAXIMA),
        ]
        .pipe(FindPeaks.validate, lazy=True)
        .pipe(DataFrame[FindPeaks])
    )

    return fp


@pa.check_types(lazy=True)
@typechecked
def width_df_factory(
    X: DataFrame[X_Schema],
    peak_t_idx: NDArray[int64],
    peak_prom_data: PPD,
    rel_height: float,
    key_X: str,
    key_width: str,
    key_height: str,
    key_left_ips: str,
    key_right_ips: str,
) -> pd.DataFrame:
    """
    width is calculated by first identifying a height to measure the width at, calculated as:
    (peak height) - ((peak prominence) * (relative height))

    prominence is defined as the vertical distance between the peak and its lowest contour line.

    prominence is calculated by:
    - from the peak maxima, extend a line left and right until it intersects with the wlen defined window bound or the slope of a higher peak, ignoring peaks the same height.
    - within this line interval, find the minima on either side. The minima is defined as the prominance bases.
    - The larger value of the pair of minima marks the lowest contour line. prominence is then calculated as the vertical difference between the peak height and the height of the contour line.
    """

    w, h, left_ips, right_ips = signal.peak_widths(
        x=X[key_X],
        peaks=peak_t_idx,
        rel_height=rel_height,
        prominence_data=peak_prom_data,
    )

    wdf_: pd.DataFrame = pd.DataFrame().rename_axis(index="idx")

    wdf_[key_width] = w
    wdf_[key_height] = h
    wdf_[key_left_ips] = left_ips
    wdf_[key_right_ips] = right_ips
    wdf: pd.DataFrame = wdf_.reset_index(names=mp_defs.P_IDX)
    return wdf


def set_peak_map(
    X: DataFrame[X_Schema],
    fp: DataFrame[FindPeaks],
    whh: DataFrame[WHH],
    pb: DataFrame[PeakBases],
) -> PeakMapOutput:

    # TODO: seperate out the different types of measurements. There are..

    pm = (
        pd.concat(
            [
                fp,
                whh.drop([mp_defs.P_IDX], axis=1),
                pb.drop([mp_defs.P_IDX], axis=1),
            ],
            axis=1,
        )
        .pipe(pl.from_pandas)
    )  # fmt: skip

    pm_melt = (
        pm
        .melt(id_vars=[mp_defs.P_IDX], value_name=mp_defs.VALUE, variable_name=mp_defs.KEY_MSNT)
        
    )  # fmt: skip

    # contains all of the geometric points mapped by the prior signal analysis methods.

    # TODO: seperate contour line measurements from peak map measurements

    # table of the peak maxima values
    maxima = prepare_maxima_tbl(pm_melt=pm_melt)

    contour_line_bounds = prepare_contour_line_bounds_tbl(pm_melt=pm_melt, X=X)

    # widths table: the scalar width of each contour line as measured by `scipy.signal.peak_widths`.

    width_keys = [mp_defs.KEY_WIDTH_WHH, mp_defs.KEY_WIDTH_PB]

    widths = (
        pm_melt.filter(pl.col(mp_defs.KEY_MSNT).is_in(width_keys))
        .to_pandas()
        .pipe(mp_schs.Widths.validate, lazy=True)
        .pipe(DataFrame[mp_schs.Widths])
    )

    output = PeakMapOutput(
        maxima=maxima,
        contour_line_bounds=contour_line_bounds,
        widths=widths,
    )

    return output


def prepare_contour_line_bounds_tbl(
    pm_melt: pl.DataFrame,
    X: DataFrame[X_Schema],
) -> DataFrame[mp_schs.ContourLineBounds]:
    contour_line_keys = [
        mp_defs.KEY_LEFT_PROM,
        mp_defs.KEY_RIGHT_PROM,
        mp_defs.KEY_LEFT_WHH,
        mp_defs.KEY_RIGHT_WHH,
        mp_defs.KEY_LEFT_PB,
        mp_defs.KEY_RIGHT_PB,
    ]

    # a normalised table of the contour bound measurements, prominence, whh, bases.

    peak_contours = pm_melt.filter(pl.col(mp_defs.KEY_MSNT).is_in(contour_line_keys))

    contour_bounds_split = (
        peak_contours.with_columns(
            pl.col(mp_defs.KEY_MSNT)
            .str.split("_")
            .list.to_struct(n_field_strategy="max_width")
            .alias("msnt_split")
        )
        .drop(mp_defs.KEY_MSNT)
        .unnest("msnt_split")
        .rename(
            {
                "field_0": mp_defs.LOC,
                "field_1": mp_defs.KEY_MSNT,
                "value": "X_idx_output",
            }
        )
    )

    countour_bounds_with_rounded_X_idx = contour_bounds_split.with_columns(
        pl.col("X_idx_output").round(0).cast(int).alias(mp_defs.KEY_X_IDX_ROUNDED)
    )

    contour_bounds_join_X = countour_bounds_with_rounded_X_idx.join(
        X.pipe(pl.from_pandas),
        left_on=mp_defs.KEY_X_IDX_ROUNDED,
        right_on=mp_defs.X_IDX,
        how="left",
    )

    peak_contours = (
        contour_bounds_join_X
        .melt(
            id_vars=[mp_defs.P_IDX,mp_defs.LOC,mp_defs.KEY_MSNT],
            value_vars=[mp_defs.KEY_X_IDX_ROUNDED,mp_defs.X],
            variable_name=mp_defs.DIM,
            value_name=mp_defs.VALUE
            )
        .sort([mp_defs.P_IDX, mp_defs.KEY_MSNT, mp_defs.LOC, mp_defs.DIM])
        .to_pandas()
        .pipe(mp_schs.ContourLineBounds.validate, lazy=True)
        .pipe(DataFrame[mp_schs.ContourLineBounds])
    )  # fmt: skip

    return peak_contours


def prepare_maxima_tbl(
    pm_melt: pl.DataFrame,
):
    maxima = (
        pm_melt.filter(pl.col(mp_defs.KEY_MSNT).is_in([mp_defs.MAXIMA, mp_defs.X_IDX]))
        .pivot(columns=mp_defs.KEY_MSNT, values=mp_defs.VALUE, index=mp_defs.P_IDX)
        .with_columns(pl.lit(mp_defs.MAXIMA).alias(mp_defs.KEY_MSNT))
        .rename({mp_defs.MAXIMA: mp_defs.X, mp_defs.KEY_MSNT: mp_defs.LOC})
        .melt(
            id_vars=[mp_defs.P_IDX, mp_defs.LOC],
            variable_name=mp_defs.DIM,
            value_name=mp_defs.VALUE,
        )
        .to_pandas()
        .pipe(mp_schs.Maxima.validate, lazy=True)
        .pipe(DataFrame[mp_schs.Maxima])
    )

    return maxima


def cast_peak_prom_data_to_tuple_of_numpy(
    fp: DataFrame[FindPeaks],
) -> PPD:
    peak_prom_data: PPD = tuple(
        [
            fp[mp_defs.KEY_PROMINENCE].to_numpy(float),
            fp[mp_defs.KEY_LEFT_PROM].to_numpy(int),
            fp[mp_defs.KEY_RIGHT_PROM].to_numpy(int),
        ]
    )  # type: ignore
    return peak_prom_data
