import polars as pl
from numpy import float64, int64
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias, Self

import numpy as np
import pandas as pd
import pandera as pa
from matplotlib.axes import Axes as Axes
from pandera.typing import DataFrame, Series
from scipy import signal  # type: ignore

from hplc_py.hplc_py_typing.typed_dicts import FindPeaksKwargs

from hplc_py.hplc_py_typing.hplc_py_typing import (
    WHH,
    FindPeaks,
    PeakBases,
    PeakMapWide,
    X_Schema,
)


from typing import Tuple

from hplc_py.io_validation import IOValid

PPD: TypeAlias = Tuple[NDArray[float64], NDArray[int64], NDArray[int64]]


class MapPeaks(IOValid):
    def __init__(
        self,
        prominence: float = 0.01,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ):
        """
        Use to map peaks, i.e. locate maxima, whh, peak bases for a given set of user inputs. Includes plotting functionality inherited from MapPeakPlots
        """
        self._prominence = prominence
        self._wlen = wlen
        self._find_peaks_kwargs = find_peaks_kwargs

        self._X_colname: str = "X"

        self._idx_name: Literal["idx"] = "idx"
        self._p_idx_col: Literal["p_idx"] = "p_idx"
        self._p_idx_colname: Literal["X_idx"] = "X_idx"
        self._maxima_colname: Literal["maxima"] = "maxima"
        self._prom_col: Literal["prom"] = "prom"
        self._prom_lb_col: Literal["prom_left"] = "prom_left"
        self._prom_rb_col: Literal["prom_right"] = "prom_right"
        self._whh_rel_height_col: Literal["whh_rel_height"] = "whh_rel_height"
        self._whh_h_col: Literal["whh_height"] = "whh_height"
        self._whh_w_col: Literal["whh_width"] = "whh_width"
        self._whh_l_col: Literal["whh_left"] = "whh_left"
        self._whh_r_col: Literal["whh_right"] = "whh_right"
        self._pb_rel_height_col: Literal["pb_rel_height"] = "pb_rel_height"
        self._pb_h_col: Literal["pb_height"] = "pb_height"
        self._pb_w_col: Literal["pb_width"] = "pb_width"
        self._pb_l_col: Literal["pb_left"] = "pb_left"
        self._pb_r_col: Literal["pb_right"] = "pb_right"

    @pa.check_types()
    def fit(
        self,
        X: DataFrame[X_Schema],
        y=None,
    ) -> Self:
        """
        In this case, simply stores X and the timestep
        """

        self.X = X

        return self

    @pa.check_types()
    def transform(
        self,
    ) -> Self:
        """
        Map An input signal with peaks, providing peak height, prominence, and width data.

        From scipy.peak_widths docs:

        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html>

        "wlen : int, optional A window length in samples passed to peak_prominences as an optional argument for internal calculation of prominence_data. This argument is ignored if prominence_data is given." - this is only used in the initial findpeaks
        then the results are propagated to the other profilings.
        """

        fp = self._set_findpeaks(
            X=self.X,
            prominence=self._prominence,
            wlen=self._wlen,
            find_peaks_kwargs=self._find_peaks_kwargs,
        )

        peak_prom_data = self.get_peak_prom_data(
            fp,
        )

        peak_t_idx = fp[self._p_idx_colname].to_numpy(np.int64)

        whh = self.width_df_factory(
            X=self.X,
            peak_t_idx=peak_t_idx,
            peak_prom_data=peak_prom_data,
            rel_height=0.5,
            prefix="whh",
        )

        whh = DataFrame[WHH](whh)

        pb = self.width_df_factory(
            X=self.X,
            peak_t_idx=peak_t_idx,
            peak_prom_data=peak_prom_data,
            rel_height=1,
            prefix="pb",
        )

        # pb = DataFrame[PeakBases](pb)

        self.peak_map = self._set_peak_map(
            fp,
            whh,
            pb,
        )

        return self

    @pa.check_types
    def _set_findpeaks(
        self,
        X: DataFrame[X_Schema],
        prominence: float,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ) -> DataFrame[FindPeaks]:
        # 'denormalize' the prominence input to put it on the scale of amp

        if wlen:
            self._check_scalar_is_type(wlen, int, "wlen")

        if not isinstance(prominence, float):
            raise TypeError(f"expected prominance to be {float}")

        if (prominence < 0) | (prominence > 1):
            raise ValueError("expect prominence to be a float between 0 and 1")

        if not isinstance(find_peaks_kwargs, dict):
            raise TypeError("Expect find_peaks_kwargs to be a dict")

        prom_ = prominence * X.max().iat[0]

        p_idx, _dict = signal.find_peaks(
            X[self._X_colname],
            prominence=prom_,
            wlen=wlen,
            **find_peaks_kwargs,
        )

        fp = DataFrame[FindPeaks](
            X.loc[p_idx]
            .assign(**{self._p_idx_colname: p_idx, **_dict})
            .reset_index(drop=True)
            .reset_index(names=self._p_idx_col)
            .rename_axis(index=self._idx_name)
            .rename(
                {
                    "prominences": self._prom_col,
                    "left_bases": self._prom_lb_col,
                    "right_bases": self._prom_rb_col,
                    self._X_colname: self._maxima_colname,
                },
                axis=1,
            )
            .loc[
                :,
                lambda df: df.columns.drop(self._maxima_colname).insert(
                    2, self._maxima_colname
                ),
            ]
        )

        return fp

    def width_df_factory(
        self,
        X: DataFrame[X_Schema],
        peak_t_idx: NDArray[int64],
        peak_prom_data: PPD,
        rel_height: float,
        prefix: str = "width",
    ) -> pd.DataFrame:
        """
        width is calculated by first identifying a height to measure the width at, calculated as:
        (peak height) - ((peak prominance) * (relative height))

        width half height, width half height height
        measure the width at half the hieght for a better approximation of
        the latent peak

        this measurement defines the 'scale' paramter of the skewnorm distribution
        for the signal peak reconstruction

        :prefix: is used to prefix the column labels, i.e. measured at half is 'whh'
        """

        rel_h_key = prefix + "_rel_height"
        w_key = prefix + "_width"
        h_key = prefix + "_height"
        left_ips_key = prefix + "_left"
        right_ips_key = prefix + "_right"

        w, h, left_ips, right_ips = signal.peak_widths(
            X[self._X_colname],
            peak_t_idx,
            rel_height,
            peak_prom_data,
        )

        wdf_: pd.DataFrame = pd.DataFrame().rename_axis(index="idx")

        wdf_[rel_h_key] = [rel_height] * len(peak_t_idx)
        wdf_[w_key] = w
        wdf_[h_key] = h
        wdf_[left_ips_key] = left_ips
        wdf_[right_ips_key] = right_ips

        wdf: pd.DataFrame = wdf_.reset_index(names=self._p_idx_col).rename_axis(
            index=self._idx_name
        )

        return wdf

    @pa.check_types
    def _set_peak_map(
        self,
        fp: DataFrame[FindPeaks],
        whh: DataFrame[WHH],
        pb: DataFrame[PeakBases],
    ) -> DataFrame[PeakMapWide]:

        pm_ = pd.concat(
            [
                fp,
                whh.drop([self._p_idx_col], axis=1),
                pb.drop([self._p_idx_col], axis=1),
            ],
            axis=1,
        )
        # .melt(id_vars="p_idx", var_name="prop", value_name="value")

        pm = PeakMapWide.validate(pm_, lazy=True)

        return pm

    def get_peak_prom_data(
        self,
        fp: DataFrame[FindPeaks],
    ) -> PPD:
        peak_prom_data: PPD = tuple(
            [
                fp[self._prom_col].to_numpy(float64),
                fp[self._prom_lb_col].to_numpy(np.int64),
                fp[self._prom_rb_col].to_numpy(np.int64),
            ]
        )  # type: ignore
        return peak_prom_data
