import numpy as np
from scipy.optimize import least_squares, _minpack
from scipy.optimize import _minpack
from scipy.optimize._minpack_py import _lightweight_memoizer, _wrap_func
from typing import Literal, Optional, Self, Callable, Type
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
from hplc_py.common.common_schemas import X_Schema
from hplc_py.io_validation import IOValid
from hplc_py.map_windows.schemas import X_Windowed
from hplc_py.pandera_helpers import PanderaSchemaMethods
from hplc_py.skewnorms.skewnorms import _compute_skewnorm_scipy, fit_skewnorms_jax
from hplc_py.deconvolution import definitions as KeysFitReport
from hplc_py.deconvolution import kwargs as dc_kwargs
from .opt_params import DataPrepper

from hplc_py.deconvolution import schemas as dc_schs, definitions as dc_defs

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

        if self.optimizer_access_str == "jax_lsq":
            import jaxfit

            jf = jaxfit.LeastSquares()
            fit_func = fit_skewnorms_jax
            self.opt_func = jf.least_squares

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

        curve_fit_output = popt_factory(
            X=self._X_w,
            params=self._params,
            optimizer=self._optfunc,
            fit_func=self._fit_func,
            # optimizer_kwargs=self._optimizer_kwargs,
            verbose=self._verbose_flag,
        )

        self.psignals: DataFrame[dc_schs.PSignals] = construct_unmixed_signals(
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
                .select(KeysFitReport.KEY_RECON)
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
        unmixed_df.groupby(KeysFitReport.P_IDX)[KeysFitReport.KEY_UNMIXED]
        .agg(["sum", "max"])
        .rename(
            {
                "sum": KeysFitReport.area_unmixed,
                "max": KeysFitReport.KEY_MAXIMA_UNMIXED,
            },
            axis=1,
            errors="raise",
        )
    )

    peak_report = (
        popt.set_index(KeysFitReport.P_IDX)
        .join(unmixed_mst)
        .reset_index()
        .loc[
            :,
            [
                KeysFitReport.W_IDX,
                KeysFitReport.P_IDX,
                KeysFitReport.MAXIMA,
                KeysFitReport.LOC,
                KeysFitReport.SCALE,
                KeysFitReport.KEY_SKEW,
                KeysFitReport.area_unmixed,
                KeysFitReport.KEY_MAXIMA_UNMIXED,
            ],
        ]
        .pipe(dc_schs.PReport.validate, lazy=True)
        .pipe(DataFrame[dc_schs.PReport])
    )

    return peak_report


def popt_factory(
    X,
    params,
    optimizer,
    fit_func,
    x_key: str,
    # optimizer_kwargs: dc_kwargs.CurveFitKwargs=dc_kwargs.curve_fit_kwargs_defaults,
    max_nfev=1e6,
    n_interms: int = 1,
    verbose: bool = True,
    terminate_on_fit: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    nfev is calculated as 100 * n by default.
    """

    if not fit_func:
        raise AttributeError(f"Please provide a fit_func, got {type(fit_func)}")

    if x_key not in X.columns:
        raise AttributeError(
            "Please provide a valid column key for 'x_key' corresponding to the observation time column"
        )

    if not optimizer:
        raise AttributeError(f"Please provide a optimizer, got {type(optimizer)}")

        # input validation

    if not int(n_interms) == n_interms:
        raise AttributeError(
            "Please provide a natural number as the n_interms argument"
        )

    if not int(max_nfev) == max_nfev:
        raise AttributeError("Please provide a natural number as the max_nfev argument")

    curve_fit_output = get_popt(
        X=X,
        params=params,
        optimizer=optimizer,
        fit_func=fit_func,
        x_key=x_key,
        max_nfev=max_nfev,
        n_interms=n_interms,
        terminate_on_fit=terminate_on_fit,
        verbose=verbose,
    )

    """
    The next and only use of the popt table is to construct the peak signals, so not too hard to fix should these outputs change - note: will have to again move all of the iteration to the pipeline class rather than here.
    """

    return curve_fit_output


def get_popt(
    X: pl.DataFrame,
    params: pl.DataFrame,
    optimizer: Callable,
    fit_func: Callable,
    x_key: str,
    max_nfev: int,
    n_interms: int,
    verbose: bool,
    terminate_on_fit: bool,
):
    """
    :param n_interms: the number of times to evaluate least squares minimization.

     Warning: `n_interms` is used to divide `max_nfev`, sharing the resulting nfev equally between each fit
     attempt, thus the ratio between `n_interms` and `max_nfev` must be large enough to arrive at a
     successful fit, otherwise the fit will fail.
    """

    # df keys. TODO: convert to ENUM
    success = "success"
    param = "param"
    p_idx = "p_idx"
    lb = "lb"
    ub = "ub"
    p0_key = "p0"
    arg_input = "arg_input"
    value = "value"
    nfev_index_key = "nfev_index"
    p_idx = "p_idx"
    param = "param"
    value = "value"
    nfev_key = "nfev"
    mesg = "mesg"
    ier = "ier"
    col = "col"
    fvec = "fvec"

    # general index columns
    idx_cols = [p_idx, param]
    params_var_name = arg_input
    param_value_name = value

    p0: pl.DataFrame = params.filter(pl.col(arg_input) == p0_key)
    bounds: pl.DataFrame = params.filter(pl.col(arg_input).is_in([lb, ub]))

    nfev_index_df = pl.DataFrame([str(-1)], schema={nfev_index_key: str})

    nfev_index_dtype = str

    idx_keys = {
        nfev_index_key: nfev_index_dtype,
        p_idx: str,
    }

    schemas = dict(
        results_df={
            **idx_keys,
            p_idx: int,
            param: str,
            "p0": float,
        },
        info_df={
            **idx_keys,
            nfev_key: int,
            mesg: str,
            ier: int,
            success: bool,
        },
        pcov_df={
            **idx_keys,
            col: str,
            value: float,
        },
        fvec_df={
            **idx_keys,
            fvec: float,
        },
    )

    intermediate_schemas = {
        k: {kk: vv for kk, vv in v.items() if kk != nfev_index_key}
        for k, v in schemas.items()
    }

    output = {k: pl.DataFrame(schema=schema) for k, schema in schemas.items()}

    # container for initial p0 and intermediate popts, all designated as p0. the actual p0 will be indexed as -1
    p0 = (
        nfev_index_df.join(p0, how="cross")
        .pivot(index=[nfev_index_key] + idx_cols, columns=arg_input, values=value)
        .drop(arg_input)
    )
    # only the results table is indexed at -1 to indicate the initial guess
    output["results_df"] = p0

    nfev_it = [int(max_nfev / n_interms)] * n_interms

    pb_desc = "nfev iter"
    colour = "green"
    verbose = True
    wrapped_it = tqdm.tqdm(
        nfev_it,
        desc=pb_desc,
        colour=colour,
        disable=not verbose,
    )

    for nfev_idx, nfev in enumerate(wrapped_it):

        input_params = (
            bounds.pivot(
                index=idx_cols,
                columns=params_var_name,
                values=param_value_name,
            )
            .with_columns(pl.lit(nfev_idx - 1).cast(str).alias(nfev_index_key))
            .join(
                output["results_df"],
                on=idx_cols + [nfev_index_key],
            )
            .select(pl.col(idx_cols), pl.col([lb, p0_key, ub]))
        )

        interm_dfs, success = curve_fit_(
            optimizer=optimizer,
            fit_func=fit_func,
            x_key=x_key,
            max_nfev=nfev,
            X=X,
            params=input_params,
            schemas=intermediate_schemas,
            verbose=verbose,
        )

        # test for empty output - indicates an error
        empty_df_bool = {k: df.is_empty() for k, df in interm_dfs.items()}

        if any(empty_df_bool.values()):
            empty_str = "\n".join(
                [f"{k} is empty" for k, v in empty_df_bool.items() if v]
            )
            error_str = f"\n{empty_str}\n\nempty dataframes detected, terminating."
            raise AttributeError(error_str)

        # index the output to the current nfev_idx
        nfev_idx_df = pl.DataFrame(
            [str(nfev_idx)], schema={nfev_index_key: nfev_index_dtype}
        )

        interm_dfs_ = {
            k: nfev_idx_df.join(df, how="cross") for k, df in interm_dfs.items()
        }

        # add the current output to the full record

        # note: info_df is broadcast to the index of params, but the information is only relevant to the
        # window each peak is a part of, rather than each individual window.

        output = {k: pl.concat([df, interm_dfs_[k]]) for k, df in output.items()}
        # if a successful fit is achieved, break the looping

        from hplc_py.deconvolution.analysis import Inspector

        # breakpoint()

        # inspector = Inspector(
        #     signal=X.to_pandas(),
        #     popt=interm_dfs["results_df"].to_pandas().pipe(get_wide_popt),
        #     x_unit="time",
        # )
        # inspector.plot_results()
        # if success and terminate_on_fit:
        #     break

        # breakpoint()
    # test: if terminate_on_fit, the unique values in the nfev index should match the input n_interms.

    return output


def curve_fit_(
    optimizer: Callable,
    fit_func: Callable,
    x_key: str,
    max_nfev: int,
    X: pl.DataFrame,
    params: pl.DataFrame,
    schemas: dict[str, Type],
    verbose: bool,
) -> tuple[dict[str, pl.DataFrame], bool]:
    """
    Curve fit with the given optimizer.
    """

    for df in [X, params]:
        if isinstance(df, pl.DataFrame):
            if df.is_empty():
                raise ValueError("Expected values")
        else:
            raise TypeError("Expected a polars DataFrame")

        if not isinstance(max_nfev, int):
            raise TypeError("max_nfev must be int")

    p_idx = "p_idx"
    param = "param"
    p0_key = "p0"

    dfs: dict = {k: pl.DataFrame(schema=schema) for k, schema in schemas.items()}

    p0: NDArray[float64] = params.select(KeysFitReport.KEY_P0).to_numpy().ravel()

    bounds: tuple[NDArray[float64], NDArray[float64]] = (
        params.select(KeysFitReport.KEY_LB).to_numpy().ravel(),
        params.select(KeysFitReport.KEY_UB).to_numpy().ravel(),
    )

    x: NDArray[float64] = X[x_key].to_numpy().ravel()

    y: NDArray[float64] = X["amplitude"].to_numpy().ravel()

    """
    To implement least_squares ala curve_fit need to implement the memoizer.
    """

    in_func = _lightweight_memoizer(_wrap_func(fit_func, x, y, transform=None))

    if verbose:
        verbose_level = 2

    # jax lsq call signature
    # res = optimizer(
    #     fun=in_func, x0=p0, xdata=x,ydata=y, bounds=bounds, max_nfev=max_nfev, verbose=verbose_level
    # )
    # scipy call sig

    res = least_squares(
        in_func, p0, bounds=bounds, max_nfev=max_nfev, verbose=verbose_level
    )

    # TODO: get jax least_squares working - atm it seems to be passing too many parameters to fit func.

    # scipy
    # res = least_squares(in_func, p0, bounds=bounds, max_nfev=max_nfev, verbose=verbose_level)

    # members of res are: message, success, fun, x, cost, jac, grad, optimality, active mask, nfev, njev
    # right now missing: success, x, jac, active mask, njev

    # passing the success bool out with the dicts then popping it above will allow me to stop iterations once a successful fit is achieved.

    results = res.x

    pcov = (
        res.jac
    )  # FIXME: modify this to match the curve_fit implementation, atm its just the jac
    success = res.success
    infodict = dict(nfev=res.nfev, fvec=np.asarray(res.fun), success=success)
    mesg = res.message
    ier = res.status
    result_idx_cols = [p_idx, param]

    # different index because of 'params'
    result_idx = params.select(result_idx_cols).clone()
    full_output_idx = result_idx.select(p_idx)

    info_df_ = pl.DataFrame(
        {"nfev": infodict["nfev"], "mesg": mesg, "ier": ier, "success": success}
    )

    pcov_df_ = pl.DataFrame(np.asarray(pcov)).melt(
        variable_name="col", value_name="value"
    )
    fvec_df_ = pl.DataFrame({"fvec": infodict["fvec"]})

    dfs["info_df"] = full_output_idx.join(info_df_, how="cross").unique()
    dfs["pcov_df"] = full_output_idx.join(pcov_df_, how="cross").unique()
    dfs["fvec_df"] = full_output_idx.join(fvec_df_, how="cross").unique()

    dfs["results_df"] = result_idx.with_columns(
        pl.from_numpy(results, schema={p0_key: float})
    )
    return dfs, success


def _skewnorm_signal_from_params(
    popt: DataFrame[dc_schs.ReconstructorPoptIn],
    x: Series[int],
) -> pd.DataFrame:
    param_keys = [
        KeysFitReport.MAXIMA,
        KeysFitReport.LOC,
        KeysFitReport.SCALE,
        KeysFitReport.KEY_SKEW,
    ]

    params: NDArray[float64] = (
            popt
            .pipe(pl.from_pandas)
            .select(pl.col(param_keys))
            .to_numpy()
            .ravel()
        )  # fmt: skip
    unmixed_signal: NDArray[float64] = _compute_skewnorm_scipy(x, params)

    unmixed_signal_df = (
        pl.DataFrame(data={KeysFitReport.KEY_UNMIXED: unmixed_signal})
        .with_row_index(name="idx")
        .select(
            popt[KeysFitReport.P_IDX].pipe(pl.from_pandas),
            x.pipe(pl.from_pandas),
            pl.col(KeysFitReport.KEY_UNMIXED),
        )
        .to_pandas()
    )

    # TODO: ADD VALIDATION
    return unmixed_signal_df


@pa.check_types(lazy=True)
def construct_unmixed_signals(
    X_w: DataFrame[dc_schs.ReconstructorSignalIn],
    popt: DataFrame[dc_schs.ReconstructorPoptIn],
    x_unit: str,
):
    """
    Constructs the peak signal along the length of the observations from popt. Returns
    a pandas dataframe in long form with stacked peak signals indexed by 'p_idx', the peak index.

    Columns: 'p_idx', 'x', 'unmixed'.

    TODO: remove X_w, modify to accept a 1D numpy array containing the observations. This will remove the need
    for x_unit.
    TODO: move to polars.
    """

    if x_unit not in X_w.columns:
        raise AttributeError("Provided `x_unit` key is not in X_w")

    x: Series[int] = Series[int](X_w[x_unit])

    peak_signals = popt.groupby(
        by=KeysFitReport.P_IDX,
        group_keys=False,
    ).apply(
        _skewnorm_signal_from_params, x
    )  # type: ignore
    return peak_signals


def reconstruct_signal(
    peak_signals: DataFrame[dc_schs.PSignals],
    x_unit: str,
):

    recon = (
        peak_signals.pipe(pl.from_pandas)
        .pivot(
            columns=KeysFitReport.P_IDX,
            index=x_unit,
            values=KeysFitReport.KEY_UNMIXED,
        )
        .select(
            pl.col(x_unit),
            pl.sum_horizontal(pl.exclude([x_unit])).alias(KeysFitReport.KEY_RECON),
        )
        .to_pandas()
        # .pipe(dc_schs.RSignal.validate, lazy=True)
        # .pipe(DataFrame[dc_schs.RSignal])
    )

    return recon


@pa.check_types(lazy=True)
def get_active_signal_as_mixed(
    tbl_signals: dc_schs.TblSignalMixed,
    x_unit: str,
    active_signal: str,
    keys: dc_defs.KeysTblMixedSignal,
) -> DataFrame[dc_schs.ActiveSignal]:
    """
    FIXME: probably can delete this as soon as the Analyzer class is gtg.

    TODO: define input schema

    this function is expecting a table in the following format:

    columns: w_type, w_idx, x_unit, signal, amplitude.

    the signal column will contain the column names post pivot.

    Define a dynamic schema using the object model to validate on input.
    """

    index = [keys.W_TYPE, keys.W_IDX, x_unit]
    columns = keys.SIGNAL
    values = keys.AMPLITUDE

    windowed_recon = (
        tbl_signals.filter(pl.col(keys.SIGNAL).is_in([active_signal, keys.RECON]))
        .pivot(
            index=index,
            columns=columns,
            values=values,
        )
        .rename(
            {
                active_signal: keys.MIXED,
            }
        )
        .to_pandas()
        .pipe(dc_schs.ActiveSignal.validate, lazy=True)
        .pipe(DataFrame[dc_schs.ActiveSignal])
    )

    return windowed_recon


def select_popt(opt_results: pl.DataFrame):
    """
    Select the last popt values calculated.
    """
    popt = opt_results.filter(
        pl.col("nfev_index").eq(pl.col("nfev_index").cast(int).max())
    )
    return popt


def is_pandas_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError

    return df


@pa.check_types
def get_wide_popt(popt: pd.DataFrame) -> DataFrame[dc_schs.Popt]:

    popt_wide = (
        # pivot so that each parameter is a column
        popt.pipe(is_pandas_dataframe)
        .pipe(pl.from_pandas)
        .pivot(index="p_idx", columns="param", values="p0")
        .to_pandas()
        .pipe(dc_schs.ReconstructorPoptIn.validate, lazy=True)
        .pipe(DataFrame[dc_schs.Popt])
    )

    return popt_wide
