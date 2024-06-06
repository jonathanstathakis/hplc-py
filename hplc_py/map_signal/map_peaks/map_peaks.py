"""
Contains methods relevant to mapping the peaks present in a signal, relying on `scipy.signal` for calculations.

NOTE: 2024-02-23 15:53:24 the `find_peaks` method of peak detection and base detection is VERY sensitive to the baseline smoothness. For example in the assessment_chrom dataset, even though the baseline correction produces no noticable adjustment in the signal, forgoing that stage results in the peak bases to be almost the length of the signal, on several of the peaks. This is presumably due to curvature and roughness in the baseline permitting the peak window lines to extend much further than they otherwise should. As SNIP smooths and corrects at the same time, it provides both functions. In conclusion, always correct and smooth, as if the peaks/peak regions do not return to zero (the implicit baseline) the windows will be much wider than desired and makes deconvolution impractical.
"""

from numpy import float64, int64
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, TypeAlias, Self

import pandas as pd
import pandera as pa
from matplotlib.axes import Axes as Axes
from pandera.typing import DataFrame as Pandas_DataFrame
from pandera.typing.polars import DataFrame as Polars_DataFrame
from polars import exceptions as pl_exceptions
from scipy import signal
import hplc_py.map_signal.map_peaks.viz_peak_map_factory
from hplc_py.map_signal.map_peaks.peak_map_output import PeakMap
from hplc_py.map_signal.map_peaks.contour_line_bounds import ContourLineBounds
from hplc_py.map_signal.map_peaks.definitions import map_peaks_kwargs_defaults
from hplc_py.common.common_schemas import X_Schema

from hplc_py.map_signal.map_peaks.definitions import MapPeaksKwargs
from hplc_py.map_signal.map_peaks import definitions as mp_defs
from hplc_py.map_signal.map_peaks import schemas as mp_schs
from hplc_py.map_signal.map_peaks import viz_matplotlib as mp_viz
from typing import Tuple

from hplc_py.io_validation import IOValid
from .schemas import WHH, FindPeaks, PeakBases
from sklearn.base import BaseEstimator, TransformerMixin

from ..map_windows import map_windows

PPD: TypeAlias = Tuple[NDArray[float64], NDArray[int64], NDArray[int64]]
import polars as pl


class PeakMapper(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        find_peaks_kwargs: MapPeaksKwargs,
    ):
        """
        Use to map peaks, i.e. locate maxima, whh, peak bases for a given set of user inputs. Includes plotting functionality inherited from MapPeakPlots
        """
        self.find_peaks_kwargs = find_peaks_kwargs
        self._fitted = False

    @pa.check_types
    def fit(
        self,
        X: Polars_DataFrame[X_Schema],
        y=None,
    ) -> Self:
        """
        In this case, simply stores X and the timestep
        """

        if self._fitted:
            raise AttributeError("This transformer instance is already fitted")
        else:

            self._fitted = True
        return self

    @pa.check_types
    def transform(
        self,
        X: Polars_DataFrame[X_Schema],
    ):
        self.X = X
        fp, whh, pb = map_peaks(X=self.X, find_peaks_kwargs=self.find_peaks_kwargs)

        check_for_empty_df(X, fp, whh, pb)

        pm: pl.DataFrame = (
            pl.concat(
                [
                    fp,
                    whh.drop(mp_defs.P_IDX),
                    pb.drop(mp_defs.P_IDX),
                ],
                how='horizontal'
            )
        )  # fmt: skip

        pm_melt: pl.DataFrame = (
            pm
            .melt(id_vars=mp_defs.P_IDX,
                value_name=mp_defs.VALUE,
                variable_name=mp_defs.KEY_MSNT)
            
        )  # fmt: skip
        assert not pm_melt.is_empty(), breakpoint()

        # contains all of the geometric points mapped by the prior signal analysis methods.

        # TODO: seperate contour line measurements from peak map measurements

        # table of the peak maxima values
        maxima = prepare_maxima_tbl(pm_melt=pm_melt)

        # widths table: the scalar width of each contour line as measured by `scipy.signal.peak_widths`.

        width_keys = [mp_defs.KEY_WIDTH_WHH, mp_defs.KEY_WIDTH_PB]

        widths = (
            pm_melt.filter(pl.col(mp_defs.KEY_MSNT).is_in(width_keys))
            .pipe(mp_schs.Widths.validate, lazy=True)
            .pipe(Polars_DataFrame[mp_schs.Widths])
        )

        contour_line_bounds: ContourLineBounds = ContourLineBounds(pm_melt=pm_melt, X=X)

        self.peak_map_ = PeakMap(
            maxima=maxima,
            contour_line_bounds=contour_line_bounds,
            widths=widths,
            X=X,
        )

        return self.peak_map_

    def viz(
        self,
        maxima: bool = True,
        whh: bool = True,
        base: str = "pb",
    ):

        if not self._fitted:
            raise RuntimeError("Please call 'fit_transform' first")

        peak_map_plots: mp_viz.PeakMapViz = self.vizzer._draw_peak_mappings(
            X=self.X,
            peak_map=self.peak_map_,
            maxima=maxima,
            whh=whh,
            base=base,
        )

        assert peak_map_plots

        return peak_map_plots


@pa.check_types(lazy=True)
def map_peaks(
    X: Polars_DataFrame[X_Schema],
    find_peaks_kwargs: mp_defs.MapPeaksKwargs = map_peaks_kwargs_defaults,
) -> tuple:
    """
    Map An input signal with peaks, providing peak height, prominence, and width data.

    From scipy.peak_widths docs:
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html>
    "wlen : int, optional A window length in samples passed to peak_prominences as an optional argument for internal calculation of prominence_data. This argument is ignored if prominence_data is given." - this is only used in the initial findpeaks
    then the results are propagated to the other profilings.
    """

    fp, peak_prom_data = set_findpeaks(
        X=X,
        find_peaks_kwargs=find_peaks_kwargs,
    )

    peak_t_idx = fp.select(pl.col(mp_defs.IDX)).to_numpy(writable=True).ravel()

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

    if whh.is_empty():
        raise ValueError(f"whh is empty. {whh.shape}")

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

    return fp, whh, pb


@pa.check_types
def set_findpeaks(
    X: Polars_DataFrame[X_Schema],
    find_peaks_kwargs: MapPeaksKwargs,
) -> Polars_DataFrame[FindPeaks]:

    if not isinstance(find_peaks_kwargs, dict):
        raise TypeError("Expect find_peaks_kwargs to be a dict")

    # move prominence to the X scale to avoid normalizing the signal
    input_prominence = find_peaks_kwargs[mp_defs.KEY_PROMINENCE]

    find_peaks_kwargs[mp_defs.KEY_PROMINENCE] = (
        dict(find_peaks_kwargs).pop(mp_defs.KEY_PROMINENCE) * X[mp_defs.X].max()
    )

    maxima_idx, _dict = signal.find_peaks(
        x=X[mp_defs.X],
        **find_peaks_kwargs,
    )

    prominence_data = _dict["prominences"], _dict["left_bases"], _dict["right_bases"]

    maxima_idx = maxima_idx.ravel()

    fp = (
        X.filter(pl.col("idx").is_in(maxima_idx))
        .with_columns(**{mp_defs.IDX: maxima_idx, **_dict})
        .with_row_index("p_idx")
        .with_columns(pl.col("p_idx").cast(int))
        .rename(
            {
                "prominences": mp_defs.KEY_PROMINENCE,
                "left_bases": mp_defs.KEY_LEFT_PROM,
                "right_bases": mp_defs.KEY_RIGHT_PROM,
                mp_defs.X: mp_defs.MAXIMA,
            },
        )
        .pipe(FindPeaks.validate, lazy=True)
        .pipe(Polars_DataFrame[FindPeaks])
    )

    return fp, prominence_data


@pa.check_types(lazy=True)
def width_df_factory(
    X: Polars_DataFrame[X_Schema],
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

    # `peak_widths` expects peaks to be a 1d iterable

    assert isinstance(peak_t_idx, np.ndarray)
    assert peak_t_idx.ndim == 1

    w, h, left_ips, right_ips = signal.peak_widths(
        x=X.select(key_X).to_numpy(writable=True).ravel(),
        peaks=peak_t_idx,
        rel_height=rel_height,
        prominence_data=peak_prom_data,
    )

    wdf_: pl.DataFrame = pd.DataFrame().rename_axis(index="idx")

    wdf_[key_width] = w
    wdf_[key_height] = h
    wdf_[key_left_ips] = left_ips
    wdf_[key_right_ips] = right_ips

    wdf: pl.DataFrame = wdf_.reset_index(names=mp_defs.P_IDX).pipe(pl.from_pandas)

    return wdf


def check_for_empty_df(X, fp, whh, pb):
    input_dfs = {"X": X, "fp": fp, "whh": whh, "pb": pb}
    empty_dfs = [df.is_empty() for df in input_dfs.values()]

    if any(empty_dfs):
        raise ValueError(
            f"empty df input: {[b for a, b in zip(empty_dfs, input_dfs) if a]}"
        )


def prepare_maxima_tbl(
    pm_melt: pl.DataFrame,
):

    from polars.exceptions import SchemaFieldNotFoundError

    try:
        maxima = (
            pm_melt.filter(
                pl.col(mp_defs.KEY_MSNT).is_in([mp_defs.MAXIMA, mp_defs.IDX])
            )
            .pivot(columns=mp_defs.KEY_MSNT, values=mp_defs.VALUE, index=mp_defs.P_IDX)
            .with_columns(pl.lit(mp_defs.MAXIMA).alias(mp_defs.KEY_MSNT))
            .rename({mp_defs.MAXIMA: mp_defs.X, mp_defs.KEY_MSNT: mp_defs.LOC})
            .melt(
                id_vars=[mp_defs.P_IDX, mp_defs.LOC],
                variable_name=mp_defs.DIM,
                value_name=mp_defs.VALUE,
            )
            .pipe(mp_schs.Maxima.validate, lazy=True)
            .pipe(Polars_DataFrame[mp_schs.Maxima])
        )
    except SchemaFieldNotFoundError as e:
        raise
        # breakpoint()

    return maxima