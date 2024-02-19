from typing import Literal, Optional, Self
from numpy import float64
from numpy.typing import NDArray

import pandas as pd
import pandera as pa
import polars as pl
import tqdm
from pandera.typing import DataFrame, Series

from hplc_py.map_peaks.schemas import PeakMapWide

from ..common_schemas import X_Schema
from .schemas import (
    Params,
    Popt,
    PReport,
    PSignals,
    RSignal,
)

from ..map_windows.schemas import X_Windowed

from ..hplc_py_typing.typed_dicts import FindPeaksKwargs
from ..io_validation import IOValid
from ..map_windows.map_windows import MapWindows
from ..pandera_helpers import PanderaSchemaMethods
from ..skewnorms.skewnorms import _compute_skewnorm_scipy

from .prepare_popt_input import DataPrepper

from .definitions import (
    W_TYPE_KEY,
    X_KEY,
    X_IDX_KEY,
    W_IDX_KEY,
    P0_KEY,
    LB_KEY,
    P_IDX_KEY,
    UB_KEY,
    PARAM_KEY,
    WHH_WIDTH_HALF_KEY,
    WHH_WIDTH_KEY,
    SKEW_KEY,
    UNMIXED_KEY,
    RETENTION_TIME_KEY,
    AREA_UNMIXED_KEY,
    MAXIMA_UNMIXED_KEY,
    PARAM_VAL_LOC,
    PARAM_VAL_MAX,
    PARAM_VAL_SKEW,
    PARAM_VAL_WIDTH,
    RECON_KEY,
)

WhichOpt = Literal["jax", "scipy"]
WhichFitFunc = Literal["jax", "scipy"]


class OptFuncReg:
    def __init__(
        self,
        optimizer: WhichOpt,
    ):
        self.opt_func = None
        self.optimizer = optimizer

        if optimizer == "jax":
            from jaxfit import CurveFit

            self.opt_func = CurveFit().curve_fit

        elif optimizer == "scipy":
            from scipy import optimize

            self.optimizer = optimize.curve_fit

        else:
            raise ValueError(f"Please provide one of {WhichOpt}")


class FitFuncReg:
    """
    Contains the fit functions for popt_factory
    """

    def __init__(
        self,
        fit_func: WhichFitFunc,
    ):
        self.fit_func = fit_func
        self.ff = None

        import hplc_py.skewnorms.skewnorms as sk

        if fit_func == "jax":
            self.ff = sk.fit_skewnorms_jax

        elif fit_func == "scipy":
            self.ff = sk._fit_skewnorms_scipy

        else:
            raise ValueError({f"Please provide one of: {WhichOpt}"})


class PeakDeconvolver(PanderaSchemaMethods, IOValid):
    @pa.check_types
    def __init__(
        self,
        which_opt: WhichOpt = "jax",
        which_fit_func: WhichFitFunc = "jax",
        prominence: float = 0.01,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
        optimizer_kwargs: dict = {},
    ):
        self.__mw = MapWindows(
            prominence=prominence, wlen=wlen, find_peaks_kwargs=find_peaks_kwargs
        )
        self.__dp = DataPrepper()

        self._X_key = X_KEY
        self._X_idx_key = X_IDX_KEY
        self._w_idx_key = W_IDX_KEY
        self._p0_key = P0_KEY
        self._lb_key = LB_KEY
        self._ub_key = UB_KEY
        self._p_idx_key = P_IDX_KEY
        self._param_key = PARAM_KEY
        self._optimizer_kwargs = optimizer_kwargs
        self._popt_idx_key = "idx"
        self._value_key = "value"
        self._verbose = True
        self._whh_width_key = WHH_WIDTH_KEY
        self._whh_width_half_key = WHH_WIDTH_HALF_KEY
        self._skew_key = SKEW_KEY
        self._unmixed_key = UNMIXED_KEY
        self._recon_key = RECON_KEY
        self._time_key = "time"
        self._area_unmixed_key = AREA_UNMIXED_KEY
        self._maxima_unmixed_key = MAXIMA_UNMIXED_KEY
        self._retention_time_key = RETENTION_TIME_KEY
        self._w_type_key = W_TYPE_KEY
        self._param_val_loc = PARAM_VAL_LOC
        self._param_val_max = PARAM_VAL_MAX
        self._param_val_width = PARAM_VAL_WIDTH
        self._param_val_skew = PARAM_VAL_SKEW

        if which_opt not in WhichOpt.__args__:  # type: ignore
            raise ValueError(f"Please provide one of {WhichOpt} to 'optimizer' kw")

        if which_fit_func not in WhichFitFunc.__args__:  # type: ignore
            raise ValueError(f"Please provide one of {WhichFitFunc} to 'optimizer' kw")

        # atm only one scipy and one jax set of fit func / optimizers, ergo they need to be paired and not mixed. Intend to add more at a later date
        if not which_opt == which_fit_func:
            raise ValueError(
                "Please pair the fit func with its corresponding optimizer"
            )

        self._optfunc = self._get_optimizer(which_opt)
        self._fit_func = self._get_fit_func(which_fit_func)

    def fit(
        self,
        X: DataFrame[X_Schema],
        timestep: float,
        y=None,
    ) -> Self:
        self._timestep = timestep
        self.X = X
        self.__mw.fit(X=X, y=y)
        return self

    def transform(
        self,
    ) -> Self:
        """
        :optimizer: string to decide which optimizer to use. Currently only 'jax' and 'scipy' are supported.
        """

        mw_tform = self.__mw.transform()
        self._X_w: DataFrame[X_Windowed] = mw_tform.X_w
        self._pm: DataFrame[PeakMapWide] = mw_tform.mp.peak_map

        _params = (
            self.__dp.fit(
                pm=self._pm,
                X_w=self._X_w,
                X_key=self._X_key,
                X_idx_key=self._X_idx_key,
                w_idx_key=self._w_idx_key,
                w_type_key=self._w_type_key,
                p_idx_key=self._p_idx_key,
                whh_width_key=self._whh_width_key,
                time_key=self._time_key,
                timestep=self._timestep,
            )
            .transform()
            .params
        )

        popt_df: pd.DataFrame = popt_factory(
            X_w=self._X_w,
            params=_params,
            optimizer=self._optfunc,
            fit_func=self._fit_func,
            X_idx_key=self._X_idx_key,
            w_idx_key=self._w_idx_key,
            p0_key=self._p0_key,
            lb_key=self._lb_key,
            ub_key=self._ub_key,
            X_key=self._X_key,
            p_idx_key=self._p_idx_key,
            param_key=self._param_key,
            popt_idx_key=self._popt_idx_key,
            value_key=self._value_key,
            optimizer_kwargs=self._optimizer_kwargs,
            verbose=self._verbose,
        )

        recon_ = construct_peak_signals(
            X_w=self._X_w,
            popt_df=popt_df,
            maxima_key=self._param_val_max,
            loc_key=self._param_val_loc,
            width_key=self._param_val_width,
            skew_key=self._param_val_skew,
            p_idx_key=self._p_idx_key,
            X_idx_key=self._X_idx_key,
            unmixed_key=self._unmixed_key,
        )

        popt_df = DataFrame[Popt](popt_df)
        psignals = DataFrame[PSignals](recon_)

        self.popt_df = popt_df
        self.psignals = psignals

        self.recon = reconstruct_signal(
            peak_signals=psignals,
            p_idx_key=self._p_idx_key,
            unmixed_key=self._unmixed_key,
            X_idx_key=self._X_idx_key,
            recon_key=self._recon_key,
        )

        self.preport = build_peak_report(
            popt=self.popt_df,
            unmixed_df=self.psignals,
            area_unmixed_key=self._area_unmixed_key,
            maxima_unmixed_key=self._maxima_unmixed_key,
            p_idx_key=self._p_idx_key,
            loc_key=self._param_val_loc,
            amp_key=self._param_val_max,
            skew_key=self._param_val_skew,
            unmixed_key=self._unmixed_key,
            w_idx_key=self._w_idx_key,
            whh_half_key=self._param_val_width,
        )

        return self

    def _get_fit_func(self, fit_func: WhichFitFunc):
        return FitFuncReg(fit_func).ff

    def _get_optimizer(self, optimizer: WhichOpt):
        return OptFuncReg(optimizer).opt_func


def build_peak_report(
    popt: DataFrame[Popt],
    unmixed_df: DataFrame[PSignals],
    unmixed_key: str,
    area_unmixed_key: str,
    maxima_unmixed_key: str,
    p_idx_key: str,
    loc_key: str,
    w_idx_key: str,
    amp_key: str,
    whh_half_key: str,
    skew_key: str,
):
    """
    add peak area to popt_df. Peak area is defined as the sum of the amplitude arrays
    of each peak
    """

    # groupby peak idx and calculate the area as the sum of amplitudes, and the maxima
    # mst - measurement
    unmixed_mst = (
        unmixed_df.groupby(p_idx_key)[unmixed_key]
        .agg(["sum", "max"])
        .rename(
            {"sum": area_unmixed_key, "max": maxima_unmixed_key}, axis=1, errors="raise"
        )
    )

    peak_report = (
        popt.copy(deep=True)
        .set_index(p_idx_key)
        .join(unmixed_mst)
        .reset_index()
        .loc[
            :,
            [
                w_idx_key,
                p_idx_key,
                loc_key,
                amp_key,
                whh_half_key,
                skew_key,
                area_unmixed_key,
                maxima_unmixed_key,
            ],
        ]
        .pipe(PReport.validate, lazy=True)
        .pipe(DataFrame[PReport])
    )

    return peak_report


def popt_factory(
    X_w: DataFrame[X_Windowed],
    params: DataFrame[Params],
    optimizer,
    fit_func,
    X_idx_key: str,
    w_idx_key: str,
    p0_key: str,
    lb_key: str,
    ub_key: str,
    X_key: str,
    p_idx_key: str,
    param_key: str,
    popt_idx_key: str,
    value_key: str,
    optimizer_kwargs={},
    verbose=True,
) -> DataFrame[Popt]:
    popt_list = []

    X_w_pl: pl.DataFrame = X_w.pipe(pl.from_pandas)
    params_pl = params.pipe(pl.from_pandas)

    # returns a dict of dict[Any, DataFrame] where the key is the partition value.

    wdw_grpby = params_pl.partition_by(
        [w_idx_key], maintain_order=True, as_dict=True
    ).items()

    if verbose:
        windows_grpby = tqdm.tqdm(
            wdw_grpby,
            desc="deconvolving windows",
        )

    else:
        windows_grpby = wdw_grpby

    from copy import deepcopy

    wix: int
    wdw: pl.DataFrame
    for wix, wdw in windows_grpby:

        optimizer_ = deepcopy(optimizer)

        p0: NDArray[float64] = wdw.select(p0_key).to_numpy().ravel()

        bounds: tuple[NDArray[float64], NDArray[float64]] = (
            wdw.select(lb_key).to_numpy().ravel(),
            wdw.select(ub_key).to_numpy().ravel(),
        )

        x: NDArray[float64] = (
            X_w_pl.filter(pl.col(w_idx_key) == wix)[X_idx_key].to_numpy().ravel()
        )

        y: NDArray[float64] = (
            X_w_pl.filter(pl.col(w_idx_key) == wix)[X_key].to_numpy().ravel()
        )

        results = optimizer_(
            fit_func,
            x,
            y,
            p0,
            bounds=bounds,
        )

        del optimizer_
        del x
        del y
        del p0
        del bounds

        # the output of `curve_fit` appears to not match the input ordering. Could

        results_pl: pl.DataFrame = (
            wdw.select([w_idx_key, p_idx_key, param_key])
            .clone()
            .with_columns(pl.Series(name=value_key, values=results[0]))
        )
        popt_list.append(results_pl)

    popt_df: DataFrame[Popt] = (
        pl.concat(popt_list)
        .pivot(
            columns=param_key,
            index=[w_idx_key, p_idx_key],
            values=value_key,
        )
        .with_row_index(popt_idx_key)
        .to_pandas()
        .astype({popt_idx_key: int})
        .set_index(popt_idx_key)
        .pipe(Popt.validate, lazy=True)
        .pipe(DataFrame[Popt])
    )

    return popt_df


@pa.check_types
def construct_peak_signals(
    X_w: DataFrame[X_Windowed],
    popt_df: DataFrame[Popt],
    maxima_key: str,
    loc_key: str,
    width_key: str,
    skew_key: str,
    p_idx_key: str,
    X_idx_key: str,
    unmixed_key: str,
) -> DataFrame[PSignals]:

    def _construct_peak_signal(
        popt_df: DataFrame[Popt],
        X_idx: Series[int],
    ) -> pd.DataFrame:

        param_keys = [maxima_key, loc_key, width_key, skew_key]

        params: NDArray[float64] = (
            popt_df
            .pipe(pl.from_pandas)
            .select(pl.col(param_keys))
            .to_numpy()
            .ravel()
        )  # fmt: skip

        unmixed_signal: NDArray[float64] = _compute_skewnorm_scipy(X_idx, params)

        unmixed_signal_df = (
            pl.DataFrame(data={unmixed_key: unmixed_signal})
            .with_row_index(name=X_idx_key)
            .select(
                popt_df[p_idx_key].pipe(pl.from_pandas),
                X_idx.pipe(pl.from_pandas),
                pl.col(unmixed_key),
            )
            .to_pandas()
            .pipe(PSignals.validate, lazy=True)
            .pipe(DataFrame[PSignals])
        )

        return unmixed_signal_df

    X_idx: Series[int] = Series[int](X_w[X_idx_key])

    peak_signals = (
        popt_df.groupby(
            by=[p_idx_key],
            group_keys=False,
        )
        .apply(_construct_peak_signal, X_idx)  # type: ignore
        .pipe(PSignals.validate, lazy=True)
        .pipe(DataFrame[PSignals])
    )

    return peak_signals


@pa.check_types
def reconstruct_signal(
    peak_signals: DataFrame[PSignals],
    p_idx_key: str,
    X_idx_key: str,
    unmixed_key: str,
    recon_key: str,
) -> DataFrame[RSignal]:

    recon = (
        peak_signals.pipe(pl.from_pandas)
        .pivot(
            columns=p_idx_key,
            index=X_idx_key,
            values=unmixed_key,
        )
        .select(
            pl.col(X_idx_key),
            pl.sum_horizontal(pl.exclude([X_idx_key])).alias(recon_key),
        )
        .to_pandas()
        .pipe(RSignal.validate, lazy=True)
        .pipe(DataFrame[RSignal])
    )

    return recon


def get_window_bounds(df: pl.DataFrame):
    """
    A convenience method to display the window bounds. For testing, will delete or
    recycle later.
    """
    bounds = df.groupby(["w_type", "w_idx"], maintain_order=True).agg(
        start=pl.col("X_idx").first(), end=pl.col("X_idx").last()
    )

    return bounds
