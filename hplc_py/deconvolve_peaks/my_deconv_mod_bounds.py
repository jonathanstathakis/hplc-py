"""
    The overall logic of the deconvolution module is as follows:

    1. iterating through each peak window:
        1. Iterating through each peak:
            1. build initial guesses as:
                - amplitude: the peak maxima,
                - location: peak t_idx,
                - width: peak width,
                - skew: 0
            2. build default bounds as:
                - amplitude: 10% peak maxima, 1000% * peak maxima.
                - location: peak window time min, peak window time max
                - width: the timestep, half the width of the window
                - skew: between negative and positive infinity
            3. add custom bounds
            4. add peak specific bounds
        5. submit extracted values to `curve_fit`
        ...

    so we could construct new tables which consist of the initial guesses, upper bounds and lower bounds for each peak in each window, i.e.:

    | # |     table_name   | window | peak | amplitude | location | width | skew |
    | 0 |  initial guesses |   1    |   1  |     70    |    200   |   10  |   0  |

    | # | table_name | window | peak | bound |  amplitude | location | width | skew |
    | 0 |    bounds  |    1   |   1  |   lb  |      7     |    100   | 0.009 | -inf |
    | 1 |    bounds  |    1   |   1  |   ub  |     700    |    300   |  100  | +inf |

    and go from there.

    The initial guess table needs the peak idx to be labelled with windows. since they both ahve the time index, thats fine. we also need the amplitudes from signal df.

    2023-12-08 10:16:41

    This test class now contains methods pertaining to the preparation stage of the deconvolution process.
"""

import polars as pl
from dataclasses import dataclass, field
from typing import Literal, Type, Self, cast

import numpy as np
import pandas as pd
import pandera as pa
import tqdm
from numpy import float64, int64
from pandera.api.pandas.model_config import BaseConfig
from pandera.typing import DataFrame, Series

from hplc_py import P0AMP, P0SKEW, P0TIME, P0WIDTH
from hplc_py.hplc_py_typing.hplc_py_typing import (
    P0,
    BaseDF,
    Bounds,
    Params,
    Popt,
    PReport,
    PSignals,
    RSignal,
    WindowedSignal,
)
from hplc_py.io_validation import IOValid
from hplc_py.map_peaks.map_peaks import PeakMapWide
from hplc_py.skewnorms.skewnorms import _compute_skewnorm_scipy

from hplc_py.pandera_helpers import PanderaSchemaMethods

WhichOpt = Literal["jax", "scipy"]
WhichFitFunc = Literal["jax", "scipy"]


class WdwPeakMapWide(PeakMapWide):
    w_type: pd.StringDtype
    w_idx: int64

    class Config:
        ordered = False
        strict = True


class InP0(pa.DataFrameModel):
    w_idx: int64
    p_idx: int64
    amp: float64
    time: float64
    whh: float64 = pa.Field(alias="whh_width")

    class Config(BaseConfig):
        name = "in_p0"
        ordered = False
        strict = True


@dataclass
class PanderaMixin:
    def try_validate(
        self,
        df: DataFrame | pd.DataFrame,
        schema,
    ):
        df_ = pd.DataFrame()

        try:
            df_ = schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as err:
            err.add_note(err.failure_cases.to_markdown())
            err.add_note("\n\n")
            err.add_note(err.data.to_markdown())
            raise err

        out_df = DataFrame[schema](df_)

        return out_df

    def get_sch_colorder(self, sch):

        # have to convert to schema object first as if the column has a field object assignment the __schema__ attribute is uninitialized at runtime
        return list(sch.to_schema().columns.keys())


@dataclass
class DataPrepper(PanderaMixin):
    ws_sc: Type[WindowedSignal] = WindowedSignal
    ws: pd.DataFrame = field(default_factory=pd.DataFrame)

    pm_sc: Type[PeakMapWide] = PeakMapWide
    pm: pd.DataFrame = field(default_factory=pd.DataFrame)

    wpm_sc: Type[WdwPeakMapWide] = WdwPeakMapWide
    wpm: pd.DataFrame = field(default_factory=pd.DataFrame)

    inp0_sc: Type[InP0] = InP0
    in_p0: pd.DataFrame = field(default_factory=pd.DataFrame)

    p0_sc: Type[P0] = P0
    p0: pd.DataFrame = field(default_factory=pd.DataFrame)

    bds_sc: Type[Bounds] = Bounds
    bds: pd.DataFrame = field(default_factory=pd.DataFrame)

    prm_sc: Type[Bounds] = Params
    params: pd.DataFrame = field(default_factory=pd.DataFrame)

    p0_param_cats = pd.CategoricalDtype(
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
        pm,
        ws,
    ) -> DataFrame[WdwPeakMapWide]:
        wpm_ = pm.copy(deep=True)
        ws_ = ws.copy(deep=True)

        t = str(self.ws_sc.time)
        t_idx = str(self.ws_sc.t_idx)
        w_type = str(self.ws_sc.w_type)
        w_idx = str(self.ws_sc.w_idx)

        wpm_ = wpm_.reset_index().set_index(t_idx)

        ws_ = ws_[[t_idx, w_type, w_idx]].set_index(t_idx)

        wpm_ = wpm_.join(ws_, how="left", validate="1:1").reset_index()
        wpm_idx_cols = self.get_sch_colorder(self.wpm_sc)
        wpm_ = wpm_.set_index(wpm_idx_cols).reset_index().set_index(self.wpm_sc.idx)

        self.wpm = self.try_validate(wpm_, self.wpm_sc)

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
        ws: DataFrame[WindowedSignal],
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

    def _prepare_params(
        self,
        pm: DataFrame[PeakMapWide],
        ws: DataFrame[WindowedSignal],
        timestep: float64,
    ) -> DataFrame[Params]:
        wpm = self._window_peak_map(pm, ws)

        in_p0_cols = self.get_sch_colorder(self.inp0_sc)

        in_p0_ = wpm.copy(deep=True).loc[:, in_p0_cols]

        self.inp0_sc.validate(in_p0_)

        in_p0 = DataFrame[self.inp0_sc](in_p0_)

        p0 = self._p0_factory(
            in_p0,
            timestep,
        )

        bounds = self._bounds_factory(
            p0,
            ws,
            timestep,
        )

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

    def __init__(
        self,
        pm: DataFrame[PeakMapWide],
        ws: DataFrame[WindowedSignal],
        timestep: float64,
        which_opt: WhichOpt = "jax",
        which_fit_func: WhichFitFunc = "jax",
    ):
        self.__dp = DataPrepper()

        # internal schemas
        self.r_signal_sch = RSignal
        self.ws_sc: Type[WindowedSignal] = WindowedSignal
        self.pm_sc: Type[PeakMapWide] = PeakMapWide
        self.wpm_sc: Type[WdwPeakMapWide] = WdwPeakMapWide
        self.inp0_sc: Type[InP0] = InP0
        self.p0_sc: Type[P0] = P0
        self.bds_sc: Type[Bounds] = Bounds
        self.prm_sc: Type[Bounds] = Params
        self.psig_sc: Type[PSignals] = PSignals

        # input validation
        self.pm = self.pm_sc.to_schema().validate(pm)
        self.ws = self.ws_sc.to_schema().validate(ws)

        self.timestep = timestep

        if which_opt not in WhichOpt.__args__:  # type: ignore
            raise ValueError(f"Please provide one of {WhichOpt} to 'optimizer' kw")

        if which_fit_func not in WhichFitFunc.__args__:  # type: ignore
            raise ValueError(f"Please provide one of {WhichFitFunc} to 'optimizer' kw")

        # atm only one scipy and one jax set of fit func / optimizers, ergo they need to be paired and not mixed. Intend to add more at a later date
        if not which_opt == which_fit_func:
            raise ValueError(
                "Please pair the fit func with its corresponding optimizer"
            )

        self.optfunc = self.get_optimizer(which_opt)
        self.fit_func = self.get_fit_func(which_fit_func)

    def get_fit_func(self, fit_func: WhichFitFunc):
        return FitFuncReg(fit_func).ff

    def get_optimizer(self, optimizer: WhichOpt):
        return OptFuncReg(optimizer).opt_func

    def deconvolve_peaks(
        self,
    ) -> Self:
        """
        :optimizer: string to decide which optimizer to use. Currently only 'jax' and 'scipy' are supported.
        """

        # as of 2024-01-30 16:37:00 cant use @check_types on columns with regex aliases (i.e. amp/amp_corrected) as it doesnt seem to be detecting `regex=True`. Work around is to validate as a schema obj.

        # checks
        params = self.__dp._prepare_params(self.pm, self.ws, self.timestep)

        popt_df: pd.DataFrame = self._popt_factory(
            self.ws,
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
        ws: DataFrame[WindowedSignal],
        params: DataFrame[Params],
        optimizer,
        fit_func,
        optimizer_kwargs={},
        verbose=True,
    ) -> DataFrame[Popt]:
        popt_list = []

        import polars as pl

        ws_ = pl.from_pandas(ws)
        params_ = pl.from_pandas(params)

        wdw_grpby = params_.partition_by(
            "w_idx", maintain_order=True, as_dict=True
        ).items()

        if verbose:
            windows_grpby = tqdm.tqdm(
                wdw_grpby,
                desc="deconvolving windows",
            )

        else:
            windows_grpby = wdw_grpby

        # ws_ = ws.set_index('w_idx')

        from copy import deepcopy

        for wix, wdw in windows_grpby:
            optimizer_ = deepcopy(optimizer)
            p0 = wdw["p0"].to_numpy()
            bounds = (wdw["lb"].to_numpy(), wdw["ub"].to_numpy())
            x = ws_.filter(pl.col("w_idx") == wix)["time"].to_numpy()
            y = ws_.filter(pl.col("w_idx") == wix)["amp_corrected"].to_numpy()

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
                wdw.select(["w_idx", "p_idx", "param"])
                .clone()
                .with_columns(values=results[0])
            )

            popt_list.append(results_)

        popt_df_ = pl.concat(popt_list)
        popt_df_ = popt_df_.pivot(
            columns="param", index=["w_idx", "p_idx"], values="values"
        )

        popt_df = DataFrame[Popt](
            popt_df_.with_row_index("idx")
            .to_pandas()
            .astype({"idx": int64})
            .set_index("idx")
        )
        return popt_df

    # @pa.check_types
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
        ).apply(
            construct_peak_signal, time
        )  # type: ignore

        peak_signals = p_signals_.reset_index(drop=True)

        return peak_signals

    @pa.check_types
    def _reconstruct_signal(
        self, peak_signals: DataFrame[PSignals]
    ) -> DataFrame[RSignal]:

        p_idx = str(self.psig_sc.p_idx)
        t_idx = str(self.psig_sc.t_idx)
        t = str(self.psig_sc.time)
        unmx = str(self.psig_sc.amp_unmixed)

        peak_signals_: pl.DataFrame = pl.from_pandas(peak_signals)  # type: ignore

        recon_sig = peak_signals_.pivot(
            columns=p_idx, index=[t_idx, t], values=unmx
        ).select(
            pl.col(t_idx, t),
            pl.sum_horizontal(pl.exclude([t_idx, t])).alias(unmx),
        )

        recon_sig_ = recon_sig.to_pandas().rename_axis(index="idx")

        RSignal.validate(recon_sig_, lazy=True)
        recon = DataFrame[RSignal](recon_sig_)

        return recon
