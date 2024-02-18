from typing import Literal, Optional, Self

import pandas as pd
import pandera as pa
import polars as pl
import tqdm
from numpy import float64, int64
from pandera.typing import DataFrame, Series

from ..common_schemas import X_Schema
from .schemas import (
    Params,
    Popt,
    PReport,
    PSignals,
    RSignal,
)

from ..map_windows.schemas import X_Windowed

from hplc_py.hplc_py_typing.typed_dicts import FindPeaksKwargs
from hplc_py.io_validation import IOValid
from hplc_py.map_windows.map_windows import MapWindows
from hplc_py.pandera_helpers import PanderaSchemaMethods
from hplc_py.skewnorms.skewnorms import _compute_skewnorm_scipy

from .prepare_popt_input import DataPrepper

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
    ):
        self.__mw = MapWindows(
            prominence=prominence, wlen=wlen, find_peaks_kwargs=find_peaks_kwargs
        )
        self.__dp = DataPrepper()

        self.X_key = "X"
        self._X_idx_key = "X_idx"
        self._w_idx_key = "w_idx"
        self._p0_key = "p0"
        self._lb_key = "lb"
        self._ub_key = "ub"
        self._p_idx_key = "p_idx"
        self._param_key = "param"
        self._value_key = "values"
        self._popt_idx_key = "idx"

        if which_opt not in WhichOpt.__args__:  # type: ignore
            raise ValueError(f"Please provide one of {WhichOpt} to 'optimizer' kw")

        if which_fit_func not in WhichFitFunc.__args__:  # type: ignore
            raise ValueError(f"Please provide one of {WhichFitFunc} to 'optimizer' kw")

        # atm only one scipy and one jax set of fit func / optimizers, ergo they need to be paired and not mixed. Intend to add more at a later date
        if not which_opt == which_fit_func:
            raise ValueError(
                "Please pair the fit func with its corresponding optimizer"
            )

        self.optfunc = self._get_optimizer(which_opt)
        self.fit_func = self._get_fit_func(which_fit_func)

    def fit(
        self,
        X: DataFrame[X_Schema],
        timestep: float,
        y=None,
    ) -> Self:
        self.timestep = timestep
        self.X = X
        self.__mw.fit(X=X, y=y)
        return self

    def transform(
        self,
    ) -> Self:
        """
        :optimizer: string to decide which optimizer to use. Currently only 'jax' and 'scipy' are supported.
        """

        self.X_w: DataFrame[X_Windowed] = self.__mw.transform().X_w

        # checks
        params = self.__dp.transform()

        popt_df: pd.DataFrame = popt_factory(
            self.X_w,
            params,
            self.optfunc,
            self.fit_func,
        )

        time = Series[float64](self.ws[str(self.ws_sc.time)])

        recon_ = construct_peak_signals(time, popt_df)

        popt_df = DataFrame[Popt](popt_df)
        psignals = DataFrame[PSignals](recon_)

        self.popt_df = popt_df
        self.psignals = psignals
        self.rsignal = reconstruct_signal(psignals)

        rs = pl.from_pandas(self.rsignal)
        ws = pl.from_pandas(self.ws)

        self.ws = (
            ws.join(rs.select([t_idx_key, unmixed_key]), on=t_idx_key, how="left")
            .to_pandas()
            .rename_axis(index="idx")
        )
        self.preport = get_peak_report(self.popt_df, self.psignals)

        return self

    def _get_fit_func(self, fit_func: WhichFitFunc):
        return FitFuncReg(fit_func).ff

    def _get_optimizer(self, optimizer: WhichOpt):
        return OptFuncReg(optimizer).opt_func

def get_peak_report(
    popt: DataFrame[Popt],
    unmixed_df: DataFrame[PSignals],
    unmixed_key: str,
    area_unmixed_key: str,
    maxima_unmixed_key: str,
    p_idx_key: str,
    time_key: str,
    retention_time_key: str,
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

    peak_report_ = (
        popt.copy(deep=True)
        .set_index(p_idx_key)
        .join(unmixed_mst)
        .reset_index()
        # express loc as retention time
        .assign(retention_time=lambda df: df[time_key])
        .astype(
            {
                retention_time_key: float64,
            }
        )
        .loc[
            :,
            [
                w_idx_key,
                p_idx_key,
                retention_time_key,
                time_key,
                amp_key,
                whh_half_key,
                skew_key,
                area_unmixed_key,
                maxima_unmixed_key,
            ],
        ]
    )

    # PReport.validate(peak_report_, lazy=True)

    peak_report = DataFrame[PReport](peak_report_)

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

    ws_: pl.DataFrame = pl.from_pandas(X_w).with_row_index(X_idx_key)
    params_ = pl.from_pandas(params)

    wdw_grpby = params_.partition_by(
        w_idx_key, maintain_order=True, as_dict=True
    ).items()

    if verbose:
        windows_grpby = tqdm.tqdm(
            wdw_grpby,
            desc="deconvolving windows",
        )

    else:
        windows_grpby = wdw_grpby

    from copy import deepcopy

    for wix, wdw in windows_grpby:
        optimizer_ = deepcopy(optimizer)
        p0 = wdw[p0_key].to_numpy()
        bounds = (
            wdw[lb_key].to_numpy(),
            wdw[ub_key].to_numpy(),
        )
        x = ws_.filter(pl.col(w_idx_key) == wix)[
            X_idx_key
        ].to_numpy()
        y = ws_.filter(pl.col(w_idx_key) == wix)[
            X_key
        ].to_numpy()

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

        results_ = (
            wdw.select(
                [w_idx_key, p_idx_key, param_key]
            )
            .clone()
            .with_columns(values=results[0])
        )

        popt_list.append(results_)

    popt_df_ = pl.concat(popt_list)
    popt_df_ = popt_df_.pivot(
        columns=param_key,
        index=[w_idx_key, p_idx_key],
        values=value_key,
    )

    popt_df = DataFrame[Popt](
        popt_df_.with_row_index(popt_idx_key)
        .to_pandas()
        .astype({popt_idx_key: int64})
        .set_index(popt_idx_key)
    )
    return popt_df

@pa.check_types
def construct_peak_signals(
    time: Series[float64],
    popt_df: DataFrame[Popt],
    amp_key: str,
    time_key: str,
    whh_half_key: str,
    skew_key: str,
    p_idx_key: str,
    t_idx_key: str,
    unmixed_key: str,
) -> DataFrame[PSignals]:
    
    def construct_peak_signal(
        popt_df: DataFrame[Popt],
        time: Series[float64],
    ) -> pd.DataFrame:
        

        params = popt_df.loc[
            :, [amp_key, time_key, whh_half_key, skew_key]
        ].values.flatten()

        unmixed_signal = _compute_skewnorm_scipy(time, params)

        # expect that time contains the same index columns as input popt_df
        popt_ = popt_df.loc[:, [p_idx_key]]

        unmixed_signal_df = pd.merge(popt_, time, how="cross")
        unmixed_signal_df = unmixed_signal_df.assign(amp_unmixed=unmixed_signal)
        unmixed_signal_df = unmixed_signal_df.reset_index(names=t_idx_key)
        unmixed_signal_df[t_idx_key] = unmixed_signal_df[t_idx_key].astype(int64)
        unmixed_signal_df = unmixed_signal_df.loc[
            :, ["p_idx", t_idx_key, time_key, unmixed_key]
        ]

        unmixed_signal_df = unmixed_signal_df.reset_index(drop=True)

        return unmixed_signal_df

    # remove w_idx from identifier to avoid multiindexed column

    p_signals_ = popt_df.groupby(
        by=["p_idx"],
        group_keys=False,
    ).apply(
        construct_peak_signal, time
    )  # type: ignore

    peak_signals = p_signals_.reset_index(drop=True)

    return peak_signals

@pa.check_types
def reconstruct_signal(
    peak_signals: DataFrame[PSignals],
    p_idx_key: str,
    X_idx_key: str,
    time_key: str,
    unmx_values_key: str,
    
) -> DataFrame[RSignal]:
    peak_signals_: pl.DataFrame = pl.from_pandas(peak_signals)  # type: ignore

    recon_sig = peak_signals_.pivot(
        columns=p_idx_key, index=X_idx_key, values=unmx_values_key,
    ).select(
        pl.col(X_idx_key, time_key),
        pl.sum_horizontal(pl.exclude([X_idx_key, time_key])).alias(unmx_values_key),
    )

    recon_sig_ = recon_sig.to_pandas()

    RSignal.validate(recon_sig_, lazy=True)
    recon = DataFrame[RSignal](recon_sig_)

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
