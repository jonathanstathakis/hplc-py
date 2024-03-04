from typing import Literal, Optional, Self
from copy import deepcopy
import pandas as pd
import pandera as pa
import polars as pl
import tqdm
from cachier import cachier
from numpy import float64
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series

from hplc_py.map_peaks.schemas import PeakMap
from hplc_py.map_windows.definitions import W_IDX, W_TYPE

from hplc_py.common.caching import CACHE_PATH, custom_param_hasher
from hplc_py.common.common_schemas import X_Schema
from hplc_py.io_validation import IOValid
from hplc_py.map_windows.schemas import X_Windowed
from hplc_py.pandera_helpers import PanderaSchemaMethods
from hplc_py.skewnorms.skewnorms import _compute_skewnorm_scipy
from hplc_py.deconvolution import definitions as dc_defs
from hplc_py.deconvolution import kwargs as dc_kwargs
from .opt_params import DataPrepper

from hplc_py.deconvolution import schemas as dc_schs

WhichOpt = Literal["jax", "scipy"]
WhichFitFunc = Literal["jax", "scipy"]


class OptFuncReg:
    def __init__(
        self,
        optimizer: WhichOpt,
    ):
        self.opt_func = None
        self.optimizer_access_str = optimizer

        if self.optimizer_access_str == "jax":
            from jaxfit import CurveFit

            self.opt_func = CurveFit().curve_fit

        elif self.optimizer_access_str == "scipy":
            from scipy import optimize

            self.opt_func = optimize.curve_fit

        else:
            raise ValueError(f"Please provide one of {WhichOpt}")

        if not self.opt_func:
            raise AttributeError("Something went wrong during optfunc retrieval")


class FitFuncReg:
    """
    Contains the fit functions for popt_factory
    """

    def __init__(
        self,
        fit_func: WhichFitFunc,
    ):
        self.fit_func_key = fit_func
        self.fit_func = None

        import hplc_py.skewnorms.skewnorms as sk

        if fit_func == "jax":
            self.fit_func = sk.fit_skewnorms_jax

        elif fit_func == "scipy":
            self.fit_func = sk._fit_skewnorms_scipy

        else:
            raise ValueError({f"Please provide one of: {WhichOpt}"})


class PeakDeconvolver(PanderaSchemaMethods, IOValid):
    @pa.check_types
    def __init__(
        self,
        which_opt: WhichOpt = "jax",
        which_fit_func: WhichFitFunc = "jax",
        optimizer_kwargs: dict = {},
        verbose: bool = True,
    ):
        """
        :optimizer: string to decide which optimizer to use. Currently only 'jax' and 'scipy' are supported.
        """

        self.__dp = DataPrepper()

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

        if not self._optfunc:
            raise AttributeError("no optfunc retrieved")

        if not self._fit_func:
            raise AttributeError("no fitfunc retrieved")

        self._optimizer_kwargs = optimizer_kwargs

        self._verbose_flag = verbose

    @pa.check_types(lazy=True)
    def fit(
        self,
        X_w: DataFrame[X_Windowed],
        params: DataFrame[dc_schs.Params],
    ) -> Self:
        self._X_w = X_w
        self._params = params
        return self

    def transform(
        self,
    ) -> Self:
        """
        Deconvolve X into a series of individual peak signals.

        primary output is: `X_w_with_recon`, the windowed X and reconstructed signal.
        secondary output:
            - `preport`: a report of the properties of each deconvolved peak
            - `recon`: a dataframe of `X_idx` and `recon` columns containing the reconstructed signal.
            - `psignals`: a long dataframe of each individual peak signal stacked, repeating `X_idx` column.
            - `params`: the parameters submitted to `curve_fit.optimize` including the lower, upper bounds and initial guess of the skewnorm distribution parameters for each peak.
            - `popt`: the optimized parameters estimated in `popt_factory`
        """

        self.popt: DataFrame[dc_schs.Popt] = popt_factory(
            X_w=self._X_w,
            params=self._params,
            optimizer=self._optfunc,
            fit_func=self._fit_func,
            # optimizer_kwargs=self._optimizer_kwargs,
            verbose=self._verbose_flag,
        )

        self.psignals: DataFrame[dc_schs.PSignals] = construct_peak_signals(
            X_w=self._X_w,
            popt=self.popt,
        )

        self.recon: DataFrame[dc_schs.RSignal] = reconstruct_signal(
            peak_signals=self.psignals,
        )

        self.X_w_with_recon: DataFrame[dc_schs.X_Windowed_With_Recon] = (
            self._X_w
            .pipe(pl.from_pandas)
            .with_columns(
                self.recon.pipe(pl.from_pandas)
                .select(dc_defs.KEY_RECON)
            )
                .to_pandas()
                .pipe(DataFrame[dc_schs.X_Windowed_With_Recon])
        )  # fmt: skip

        self.preport: DataFrame[dc_schs.PReport] = build_peak_report(
            popt=self.popt,
            unmixed_df=self.psignals,
        )

        return self

    def _get_fit_func(self, fit_func: WhichFitFunc):
        return FitFuncReg(fit_func).fit_func

    def _get_optimizer(self, optimizer: WhichOpt):
        return OptFuncReg(optimizer).opt_func


def build_peak_report(
    popt: DataFrame[dc_schs.Popt],
    unmixed_df: DataFrame[dc_schs.PSignals],
):
    """
    add peak area to popt_df. Peak area is defined as the sum of the amplitude arrays
    of each peak
    """

    # groupby peak idx and calculate the area as the sum of amplitudes, and the maxima
    # mst - measurement
    unmixed_mst = (
        unmixed_df.groupby(dc_defs.P_IDX)[dc_defs.KEY_UNMIXED]
        .agg(["sum", "max"])
        .rename(
            {"sum": dc_defs.KEY_AREA_UNMIXED, "max": dc_defs.KEY_MAXIMA_UNMIXED},
            axis=1,
            errors="raise",
        )
    )

    peak_report = (
        popt.set_index(dc_defs.P_IDX)
        .join(unmixed_mst)
        .reset_index()
        .loc[
            :,
            [
                dc_defs.W_IDX,
                dc_defs.P_IDX,
                dc_defs.MAXIMA,
                dc_defs.LOC,
                dc_defs.SCALE,
                dc_defs.KEY_SKEW,
                dc_defs.KEY_AREA_UNMIXED,
                dc_defs.KEY_MAXIMA_UNMIXED,
            ],
        ]
        .pipe(dc_schs.PReport.validate, lazy=True)
        .pipe(DataFrame[dc_schs.PReport])
    )

    return peak_report


@cachier(hash_func=custom_param_hasher, cache_dir=CACHE_PATH)
def popt_factory(
    X_w: DataFrame[X_Windowed],
    params: DataFrame[dc_schs.Params],
    optimizer,
    fit_func,
    x_key: str,
    # optimizer_kwargs: dc_kwargs.CurveFitKwargs=dc_kwargs.curve_fit_kwargs_defaults,
    max_nfev=1e6,
    verbose=True,
) -> DataFrame[dc_schs.Popt]:
    """
    TODO: modify the `curve_fit` implementation to report progress:
        - can take the max_iter, break it up into fractions, report the popt, and enter those popt into the next iteration as according to [stack overflow](https://stackoverflow.com/questions/54560909/advanced-curve-fitting-methods-that-allows-real-time-monitoring)

    nfev is calculated as 100 * n by default.
    """

    if not optimizer:
        raise AttributeError(f"Please provide a optimizer, got {type(optimizer)}")
    if not fit_func:
        raise AttributeError(f"Please provide a fit_func, got {type(fit_func)}")

    X_w_keys = X_w.columns

    if x_key not in X_w_keys:
        raise AttributeError(
            "Please provide a valid column key for 'x_key' corresponding to the observation time column"
        )

    popt_list = []

    X_w_pl: pl.DataFrame = X_w.pipe(pl.from_pandas)
    params_pl = params.pipe(pl.from_pandas)

    # returns a dict of dict[Any, DataFrame] where the key is the partition value.

    wdw_grpby = params_pl.partition_by(
        [dc_defs.W_IDX], maintain_order=True, as_dict=True
    ).items()

    if verbose:
        windows_grpby = tqdm.tqdm(
            wdw_grpby,
            desc="deconvolving windows",
        )

    else:
        windows_grpby = wdw_grpby

    wix: int
    wdw: pl.DataFrame
    for wix, wdw in windows_grpby:

        optimizer_ = deepcopy(optimizer)

        p0: NDArray[float64] = wdw.select(dc_defs.KEY_P0).to_numpy().ravel()

        bounds: tuple[NDArray[float64], NDArray[float64]] = (
            wdw.select(dc_defs.KEY_LB).to_numpy().ravel(),
            wdw.select(dc_defs.KEY_UB).to_numpy().ravel(),
        )

        x: NDArray[float64] = (
            X_w_pl.filter(pl.col(dc_defs.W_IDX) == wix)[x_key].to_numpy().ravel()
        )

        y: NDArray[float64] = (
            X_w_pl.filter(pl.col(dc_defs.W_IDX) == wix)["amplitude"].to_numpy().ravel()
        )

        results = optimizer_(
            fit_func,
            x,
            y,
            p0,
            bounds=bounds,
            max_nfev=max_nfev,
            # **optimizer_kwargs,
        )

        del optimizer_
        del x
        del y
        del p0
        del bounds

        # the output of `curve_fit` appears to not match the input ordering. Could

        results_pl: pl.DataFrame = (
            wdw.select([dc_defs.W_IDX, dc_defs.P_IDX, dc_defs.PARAM])
            .clone()
            .with_columns(pl.Series(name=dc_defs.VALUE, values=results[0]))
        )
        popt_list.append(results_pl)

    popt_df: DataFrame[dc_schs.Popt] = (
        pl.concat(popt_list)
        .pivot(
            columns=dc_defs.PARAM,
            index=[dc_defs.W_IDX, dc_defs.P_IDX],
            values=dc_defs.VALUE,
        )
        .with_row_index(dc_defs.KEY_POPT_IDX)
        .to_pandas()
        .astype({dc_defs.KEY_POPT_IDX: int})
        .set_index(dc_defs.KEY_POPT_IDX)
        .pipe(dc_schs.Popt.validate, lazy=True)
        .pipe(DataFrame[dc_schs.Popt])
    )

    return popt_df


def _skewnorm_signal_from_params(
    popt: DataFrame[dc_schs.Popt],
    x: Series[int],
) -> pd.DataFrame:

    param_keys = [dc_defs.MAXIMA, dc_defs.LOC, dc_defs.SCALE, dc_defs.KEY_SKEW]

    params: NDArray[float64] = (
            popt
            .pipe(pl.from_pandas)
            .select(pl.col(param_keys))
            .to_numpy()
            .ravel()
        )  # fmt: skip
    unmixed_signal: NDArray[float64] = _compute_skewnorm_scipy(x, params)

    unmixed_signal_df = (
        pl.DataFrame(data={dc_defs.KEY_UNMIXED: unmixed_signal})
        .with_row_index(name="idx")
        .select(
            popt[dc_defs.P_IDX].pipe(pl.from_pandas),
            x.pipe(pl.from_pandas),
            pl.col(dc_defs.KEY_UNMIXED),
        )
        .to_pandas()
    )

    # TODO: ADD VALIDATION
    return unmixed_signal_df


def construct_peak_signals(
    X_w: DataFrame[X_Windowed],
    popt: DataFrame[dc_schs.Popt],
    x_unit: str,
):

    if x_unit not in X_w.columns:
        raise AttributeError("Provided `x_unit` key is not in X_w")

    x: Series[int] = Series[int](X_w[x_unit])

    peak_signals = (
        popt.groupby(
            by=[dc_defs.W_IDX, dc_defs.P_IDX],
            group_keys=False,
        )
        .apply(_skewnorm_signal_from_params, x)  # type: ignore
    )
    return peak_signals

def reconstruct_signal(
    peak_signals: DataFrame[dc_schs.PSignals],
    x_unit: str,
):

    recon = (
        peak_signals.pipe(pl.from_pandas)
        .pivot(
            columns=dc_defs.P_IDX,
            index=x_unit,
            values=dc_defs.KEY_UNMIXED,
        )
        .select(
            pl.col(x_unit),
            pl.sum_horizontal(pl.exclude([x_unit])).alias(dc_defs.KEY_RECON),
        )
        .to_pandas()
        # .pipe(dc_schs.RSignal.validate, lazy=True)
        # .pipe(DataFrame[dc_schs.RSignal])
    )

    return recon
