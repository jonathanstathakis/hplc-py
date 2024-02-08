from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Self, Type, cast

import numpy as np
import pandas as pd
import pandera as pa
import polars as pl
import tqdm
from numpy import float64, int64
from pandera.api.pandas.model_config import BaseConfig
from pandera.typing import DataFrame, Series

from hplc_py import P0AMP, P0SKEW, P0TIME, P0WIDTH
from hplc_py.hplc_py_typing.hplc_py_typing import (
    P0,
    Bounds,
    InP0,
    Params,
    Popt,
    PReport,
    PSignals,
    RSignal,
    WdwPeakMap,
    X_Schema,
    X_Windowed,
)
from hplc_py.hplc_py_typing.typed_dicts import FindPeaksKwargs
from hplc_py.io_validation import IOValid
from hplc_py.map_signals.map_peaks.map_peaks import MapPeaks, PeakMap
from hplc_py.map_signals.map_windows import MapWindows
from hplc_py.pandera_helpers import PanderaSchemaMethods
from hplc_py.skewnorms.skewnorms import _compute_skewnorm_scipy

WhichOpt = Literal["jax", "scipy"]
WhichFitFunc = Literal["jax", "scipy"]

def get_window_bounds(
    df: pl.DataFrame):
    """
    A convenience method to display the window bounds. For testing, will delete or 
    recycle later.
    """
    bounds = df.groupby(["w_type", "w_idx"], maintain_order=True).agg(
        start=pl.col("X_idx").first(), end=pl.col("X_idx").last()
    )

    return bounds


class DataPrepper:
    def __init__(self):
        self.ws: pd.DataFrame = DataFrame()
        self.wpm: pd.DataFrame = DataFrame()

        self.in_p0: pd.DataFrame = DataFrame()
        self.p0: pd.DataFrame = DataFrame()

        self.bds: pd.DataFrame = DataFrame()

        self.params: pd.DataFrame = DataFrame()

        self.p0_param_cats = pd.CategoricalDtype(
            [
                P0AMP,
                P0TIME,
                P0WIDTH,
                P0SKEW,
            ],
            ordered=True,
        )

    """
    class containing methods to prepare the data for deconvolution
    """

    def _window_peak_map(
        self,
        pm: DataFrame[PeakMap],
        X_w: DataFrame[X_Windowed],
    ) -> DataFrame[WdwPeakMap]:
        """
        add w_idx to to peak map for later lookups
        """

        pm_ = pm.rename({"t_idx": "X_idx"}, axis=1)
        X_w_ = X_w.reset_index(names="X_idx")[["w_type", "w_idx", "X_idx"]]

        X_w__ = pl.from_pandas(X_w_)
        pm__ = pl.from_pandas(pm_)

        wpm = pm__.join(X_w__, how="left", on="X_idx", validate="1:1").select(
            pl.col(["w_type", "w_idx"]), pl.exclude(["w_type", "w_idx"])
        )

        window_bounds = X_w__.pipe(get_window_bounds)
        breakpoint()
        if wpm.select(pl.col("w_type").is_in(["interpeak"]).any()).item():
            raise ValueError("peak has been assigned to interpeak region.")

        breakpoint()

        return self.wpm

    def _p0_factory(
        self,
        wpm: DataFrame[InP0],
        timestep: float64,
    ) -> DataFrame[P0]:
        """
        Build a table of initial guesses for each peak in each window.
        """
        # window the peak map

        p0_ = wpm.copy(deep=True)

        # assign skew as zero as per definition
        p0_[P0SKEW] = pd.Series([0.0] * len(p0_), dtype=float64)

        # assign whh as half peak whh as per definition, in time units
        p0_[P0WIDTH] = p0_.pop(str(self.inp0_sc.whh)).div(2).mul(float(timestep))

        # set index as idx, w_idx, p_idx
        p0_ = p0_.set_index(
            [str(self.inp0_sc.w_idx), str(self.inp0_sc.p_idx)], append=True
        )

        # go from wide to long with index as above + a param col, 1 value col p0
        p0_ = (
            p0_.stack()
            .reset_index(level=3)
            .rename(
                {"level_3": str(self.p0_sc.param), 0: str(self.p0_sc.p0)},
                axis=1,
                errors="raise",
            )
        )

        # set the param col as an ordered categorical
        p0_[str(self.p0_sc.param)] = pd.Categorical(
            p0_[str(self.p0_sc.param)], dtype=self.p0_param_cats
        )

        # add the param col to index and sort
        p0_ = p0_.set_index(str(self.p0_sc.param), append=True).sort_index()

        # return index cols to columns
        p0_ = p0_.reset_index(
            [str(self.p0_sc.w_idx), str(self.p0_sc.p_idx), str(self.p0_sc.param)]
        )

        # reset to range index
        p0_ = p0_.reset_index(drop=True).rename_axis(index=str(self.p0_sc.idx))

        p0 = self.try_validate(p0_, self.p0_sc)

        self.p0 = p0

        return p0

    def _bounds_factory(
        self,
        p0: DataFrame[P0],
        ws: DataFrame[X_Windowed],
        timestep: float64,
    ) -> DataFrame[Bounds]:
        """
        Build a default bounds df from the `signal_df`, `peak_df`, and `window_df` in the following format:

        | # | table_name | window | p_idx | bound |  amp | location | width | skew |
        |---|------------|--------|----------|-------|------|----------|-------|------|
        | 0 |    bounds  |    1   |     1    |   lb  |   7  |    100   | 0.009 | -inf |
        | 1 |    bounds  |    1   |     1    |   ub  |  70  |    300   |  100  | +inf |

        The bounds are defined as follows:

        | parameter | bound | formula                          |
        |-----------|-------|----------------------------------|
        | amplitude |   lb  | 10% peak maxima                  |
        | amplitude |   ub  | 1000% peak maxima                |
        | location  |   lb  | minimum time index of the window |
        | location  |   ub  | maximum time index of the window |
        | width     |   lb  | magnitude of the timestep        |
        | width     |   ub  | half the width of the window     |
        | skew      |   lb  | -inf                             |
        | skew      |   ub  | +inf                             |

        - amplitude: 10% peak maxima, 1000% * peak maxima.
                - location: peak window time min, peak window time max
                - width: the timestep, half the range of the window
                - skew: between negative and positive infinity
        """

        timestep = float64(timestep)

        bounds_ = pd.DataFrame(
            p0.loc[:, [self.p0_sc.w_idx, self.p0_sc.p_idx, self.p0_sc.param]],
            index=p0.index,
        )
        bounds_[self.bds_sc.lb] = pd.Series([np.nan * len(p0)], dtype=float64)
        bounds_[self.bds_sc.ub] = pd.Series([np.nan * len(p0)], dtype=float64)

        bounds_ = bounds_.set_index(
            [self.bds_sc.w_idx, self.bds_sc.p_idx, self.bds_sc.param]
        )

        # amp
        amp = p0.set_index(["param"]).loc["amp"].reset_index()
        amp = (
            amp.groupby(["w_idx", "p_idx", "param"], observed=True)["p0"]
            .agg([("lb", lambda x: x * 0.1), ("ub", lambda x: x * 10)])  # type: ignore
            .dropna()
        )
        # loc

        bounds_.loc[amp.index, [self.bds_sc.lb, self.bds_sc.ub]] = amp

        loc_b = (
            ws.loc[ws[self.ws_sc.w_type] == "peak"]
            .groupby(str(self.ws_sc.w_idx))[str(self.ws_sc.time)]
            .agg([("lb", "min"), ("ub", "max")])  # type: ignore
            .reset_index()
            .assign(param=P0TIME)
            .set_index([str(self.bds_sc.w_idx), str(self.bds_sc.param)])
        )

        # get the peak idx from p0
        loc_b = (
            p0.set_index(["w_idx", "param"])
            .drop("p0", axis=1)
            .join(loc_b.reset_index().set_index(["w_idx", "param"]), how="right")
            .reset_index()
        )

        loc_b = loc_b.set_index(["w_idx", "p_idx", "param"])
        bounds_.loc[loc_b.index, [self.bds_sc.lb, self.bds_sc.ub]] = loc_b

        bounds_.loc[pd.IndexSlice[:, :, P0WIDTH], str(self.bds_sc.lb)] = timestep  # type: ignore

        width_ub = (
            ws.loc[ws[self.ws_sc.w_type] == "peak"]
            .groupby(str(self.ws_sc.w_idx))[str(self.ws_sc.time)]
            .agg(lambda x: (x.max() - x.min()) / 2)
            .rename(self.bds_sc.ub)
            .reset_index()
            .assign(param=P0WIDTH)
            .set_index([self.ws_sc.w_idx, self.bds_sc.param])
        )

        width_ub = (
            p0.set_index(["w_idx", "param"])
            .drop("p0", axis=1)
            .join(width_ub.reset_index().set_index(["w_idx", "param"]), how="right")
            .reset_index()
            .set_index(["w_idx", "p_idx", "param"])
        )

        bounds_.loc[width_ub.index, str(self.bds_sc.ub)] = width_ub

        # skew

        bounds_.loc[pd.IndexSlice[:, :, P0SKEW], self.bds_sc.lb] = -np.inf
        bounds_.loc[pd.IndexSlice[:, :, P0SKEW], self.bds_sc.ub] = np.inf

        column_ordering = self.get_sch_colorder(self.bds_sc)

        bounds_ = bounds_.reset_index()
        bounds_ = bounds_.reindex(column_ordering, axis=1).rename_axis(
            index=self.bds_sc.idx
        )

        self.bounds = self.try_validate(bounds_, self.bds_sc)

        return self.bounds

    def transform(
        self,
        pm: DataFrame[PeakMap],
        X_w: DataFrame[X_Windowed],
        timestep: float64,
    ) -> DataFrame[Params]:
        """
        Prepare the parameter input to `optimize`, i.e. the lb, p0 and ub for each parameter
        of the skewnorm model.

        :param pm: peakmap table
        :type pm: DataFrame[PeakMap]
        :param ws: windowed signal table
        :type ws: DataFrame[X_Windowed]
        :param timestep: the timestep
        :type timestep: float64
        :return: the parameter table in long form with the 4 parameters of each peak of
        each window
        :rtype: DataFrame[Params]
        """
        wpm = self._window_peak_map(pm, X_w)

        # the input to `p0_factory` is depicted in the In_p0 schema. Use it to subset
        # the wpm then submit to to `p0_factory`

        breakpoint()
        in_p0_ = wpm.copy(deep=True).loc[:, in_p0_cols]
        self.inp0_sc.validate(in_p0_)
        in_p0 = DataFrame[self.inp0_sc](in_p0_)

        p0 = self._p0_factory(
            in_p0,
            timestep,
        )

        bounds = self._bounds_factory(
            p0,
            X_w,
            timestep,
        )

        # join the p0 and bounds tables

        join_cols = [self.p0_sc.w_idx, self.p0_sc.p_idx, self.p0_sc.param]

        p0_ = p0.reset_index().set_index(join_cols)
        bounds_ = bounds.set_index(join_cols)

        params_ = p0_.join(bounds_, how="left", validate="1:1")
        params_ = params_.reset_index().set_index(self.prm_sc.idx)

        self.params = self.try_validate(params_, self.prm_sc)

        return self.params


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

        self.X_colname = "X"
        self._X_idx_colname = "X_idx"
        self._w_idx_colname = "w_idx"
        self._p0_colname = "p0"
        self._lb_colname = "lb"
        self._ub_colname = "ub"
        self._p_idx_colname = "p_idx"
        self._param_colname = "param"
        self._value_colname = "values"
        self._popt_idx_colname = "idx"

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
        self.__mw.fit(X=X, timestep=timestep, y=y)
        return self

    def transform(
        self,
    ) -> Self:
        """
        :optimizer: string to decide which optimizer to use. Currently only 'jax' and 'scipy' are supported.
        """

        self.X_w: DataFrame[X_Windowed] = self.__mw.transform().X_w

        # checks
        params = self.__dp.transform(self.__mw.pm.peak_map, self.X_w, self.timestep)

        popt_df: pd.DataFrame = self._popt_factory(
            self.X_w,
            params,
            self.optfunc,
            self.fit_func,
        )

        time = Series[float64](self.ws[str(self.ws_sc.time)])

        recon_ = self._construct_peak_signals(time, popt_df)

        popt_df = DataFrame[Popt](popt_df)
        psignals = DataFrame[PSignals](recon_)

        self.popt_df = popt_df
        self.psignals = psignals
        self.rsignal = self._reconstruct_signal(psignals)

        rs = pl.from_pandas(self.rsignal)
        ws = pl.from_pandas(self.ws)

        self.ws = (
            ws.join(rs.select(["t_idx", "amp_unmixed"]), on="t_idx", how="left")
            .to_pandas()
            .rename_axis(index="idx")
        )
        self.preport = self._get_peak_report(self.popt_df, self.psignals)

        return self

    def _get_fit_func(self, fit_func: WhichFitFunc):
        return FitFuncReg(fit_func).ff

    def _get_optimizer(self, optimizer: WhichOpt):
        return OptFuncReg(optimizer).opt_func

    def _get_peak_report(
        self,
        popt: DataFrame[Popt],
        unmixed_df: DataFrame[PSignals],
    ):
        """
        add peak area to popt_df. Peak area is defined as the sum of the amplitude arrays
        of each peak
        """

        # groupby peak idx and calculate the area as the sum of amplitudes, and the maxima
        # mst - measurement
        unmixed_mst = (
            unmixed_df.groupby("p_idx")["amp_unmixed"]
            .agg(["sum", "max"])
            .rename(
                {"sum": "area_unmixed", "max": "maxima_unmixed"}, axis=1, errors="raise"
            )
        )

        peak_report_ = (
            popt.copy(deep=True)
            .set_index("p_idx")
            .join(unmixed_mst)
            .reset_index()
            .rename_axis(index="idx")
            # express loc as retention time
            .assign(retention_time=lambda df: df["time"])
            .astype(
                {
                    "retention_time": float64,
                }
            )
            .loc[
                :,
                [
                    "w_idx",
                    "p_idx",
                    "retention_time",
                    "time",
                    "amp",
                    "whh_half",
                    "skew",
                    "area_unmixed",
                    "maxima_unmixed",
                ],
            ]
        )

        # PReport.validate(peak_report_, lazy=True)

        peak_report = DataFrame[PReport](peak_report_)

        return peak_report

    def _popt_factory(
        self,
        X_w: DataFrame[X_Windowed],
        params: DataFrame[Params],
        optimizer,
        fit_func,
        optimizer_kwargs={},
        verbose=True,
    ) -> DataFrame[Popt]:
        popt_list = []

        ws_ = pl.from_pandas(X_w).with_row_index(self._X_idx_colname)
        params_ = pl.from_pandas(params)

        wdw_grpby = params_.partition_by(
            self._w_idx_colname, maintain_order=True, as_dict=True
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
            p0 = wdw[self._p0_colname].to_numpy()
            bounds = (
                wdw[self._lb_colname].to_numpy(),
                wdw[self._ub_colname].to_numpy(),
            )
            x = ws_.filter(pl.col(self._w_idx_colname) == wix)[
                self._X_idx_colname
            ].to_numpy()
            y = ws_.filter(pl.col(self._w_idx_colname) == wix)[
                self.X_colname
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
                    [self._w_idx_colname, self._p_idx_colname, self._param_colname]
                )
                .clone()
                .with_columns(values=results[0])
            )

            popt_list.append(results_)

        popt_df_ = pl.concat(popt_list)
        popt_df_ = popt_df_.pivot(
            columns=self._param_colname,
            index=[self._w_idx_colname, self._p_idx_colname],
            values=self._value_colname,
        )

        popt_df = DataFrame[Popt](
            popt_df_.with_row_index(self._popt_idx_colname)
            .to_pandas()
            .astype({self._popt_idx_colname: int64})
            .set_index(self._popt_idx_colname)
        )
        return popt_df

    @pa.check_types
    def _construct_peak_signals(
        self,
        time: Series[float64],
        popt_df: DataFrame[Popt],
    ) -> DataFrame[PSignals]:
        def construct_peak_signal(
            popt_df: DataFrame[Popt],
            time: Series[float64],
        ) -> pd.DataFrame:
            self.check_container_is_type(time, pd.Series, float64)

            params = popt_df.loc[
                :, ["amp", "time", "whh_half", "skew"]
            ].values.flatten()

            unmixed_signal = _compute_skewnorm_scipy(time, params)

            # expect that time contains the same index columns as input popt_df
            popt_ = popt_df.loc[:, ["p_idx"]]

            unmixed_signal_df = pd.merge(popt_, time, how="cross")
            unmixed_signal_df = unmixed_signal_df.assign(amp_unmixed=unmixed_signal)
            unmixed_signal_df = unmixed_signal_df.reset_index(names="t_idx")
            unmixed_signal_df["t_idx"] = unmixed_signal_df["t_idx"].astype(int64)
            unmixed_signal_df = unmixed_signal_df.loc[
                :, ["p_idx", "t_idx", "time", "amp_unmixed"]
            ]

            unmixed_signal_df = unmixed_signal_df.reset_index(drop=True)

            return unmixed_signal_df

        # remove w_idx from identifier to avoid multiindexed column

        p_signals_ = popt_df.groupby(
            by=["p_idx"],
            group_keys=False,
        ).apply(construct_peak_signal, time)  # type: ignore

        peak_signals = p_signals_.reset_index(drop=True)

        return peak_signals

    @pa.check_types
    def _reconstruct_signal(
        self, peak_signals: DataFrame[PSignals]
    ) -> DataFrame[RSignal]:
        peak_signals_: pl.DataFrame = pl.from_pandas(peak_signals)  # type: ignore

        breakpoint()
        recon_sig = peak_signals_.pivot(
            columns=self._p_idx_colname, index=self._X_idx_colname, values=unmx
        ).select(
            pl.col(t_idx, t),
            pl.sum_horizontal(pl.exclude([t_idx, t])).alias(unmx),
        )

        recon_sig_ = recon_sig.to_pandas().rename_axis(index="idx")

        RSignal.validate(recon_sig_, lazy=True)
        recon = DataFrame[RSignal](recon_sig_)

        return recon
