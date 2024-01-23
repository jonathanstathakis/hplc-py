"""
    The overall logic of the deconvolution module is as follows:

    1. iterating through each peak window:
        1. Iterating through each peak:
            1. build initial guesses as:
                - amplitude: the peak maxima,
                - location: peak time_idx,
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

from dataclasses import dataclass, field
from typing import Type

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pt
import tqdm
from pandera.typing import DataFrame
from scipy import optimize
from hplc_py.skewnorms.skewnorms import _compute_skewnorm

from hplc_py import P0AMP, P0SKEW, P0TIME, P0WIDTH
from hplc_py.hplc_py_typing.hplc_py_typing import (
    P0,
    BaseDF,
    Bounds,
    FloatArray,
    PReport,
    OutWindowDF_Base,
    Params,
    Popt,
    PSignals,
    SignalDF,
)
from hplc_py.map_signals.map_peaks import PeakMap
from hplc_py.map_signals.map_windows import WindowedSignal
from hplc_py.hplc_py_typing.hplc_py_typing import RSignal

class WdwPeakMap(PeakMap):
    w_type: pd.StringDtype
    w_idx: pd.Int64Dtype

    class Config:
        ordered = False
        strict = True


from pandera.api.pandas.model_config import BaseConfig


class InP0(BaseDF):
    w_idx: pd.Int64Dtype
    p_idx: pd.Int64Dtype
    amp: pd.Float64Dtype
    time: pd.Float64Dtype
    whh: pd.Float64Dtype = pa.Field(alias="whh_width")

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

        return df_

    def get_sc_cols(
        self,
        sch,
    ):
        cols = list(sch.get_metadata()[sch.Config.name]["columns"].keys())

        return cols


@dataclass
class DataPrepper(PanderaMixin):
    ws_sc: Type[WindowedSignal] = WindowedSignal
    ws: pd.DataFrame = field(default_factory=pd.DataFrame)

    pm_sc: Type[PeakMap] = PeakMap
    pm: pd.DataFrame = field(default_factory=pd.DataFrame)

    wpm_sc: Type[WdwPeakMap] = WdwPeakMap
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
        pm: DataFrame[PeakMap],
        ws: DataFrame[WindowedSignal],
    ) -> DataFrame[WdwPeakMap]:
        wpm_ = pm.copy(deep=True)
        ws_ = ws.copy(deep=True)

        wpm_ = wpm_.reset_index().set_index(self.pm_sc.time_idx)

        ws_ = ws_.drop([self.ws_sc.time, self.ws_sc.amp], axis=1).set_index(
            self.ws_sc.time_idx
        )

        wpm_ = wpm_.join(ws_, how="left", validate="1:1").reset_index()

        wpm_idx_cols = [
            self.wpm_sc.w_type,
            self.wpm_sc.w_idx,
            self.wpm_sc.p_idx,
            self.wpm_sc.time_idx,
            self.wpm_sc.time,
        ]

        wpm_ = wpm_.set_index(wpm_idx_cols).reset_index().set_index(self.wpm_sc.idx)

        self.wpm = self.try_validate(wpm_, self.wpm_sc)

        return self.wpm

    # @pa.check_types
    def _p0_factory(
        self,
        wpm: DataFrame[InP0],
        timestep: float,
    ) -> pt.DataFrame[P0]:
        """
        Build a table of initial guesses for each peak in each window following the format:

        | # |     table_name   | window | p_idx | amplitude | location | width | skew |
        | 0 |  initial guesses |   1    |   1  |     70    |    200   |   10  |   0  |

        The initial guess for each peak is simply its maximal amplitude, the time idx of the maximal amplitude, estimated width divided by 2, and a skew of zero.
        """
        # window the peak map

        p0_ = wpm.copy(deep=True)

        # assign skew as zero as per definition
        p0_[P0SKEW] = pd.Series([0.0] * len(p0_), dtype=pd.Float64Dtype())

        # assign whh as half peak whh as per definition, in time units
        p0_[P0WIDTH] = p0_.pop(str(self.inp0_sc.whh)).div(2) * timestep

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

    @pa.check_types
    def _bounds_factory(
        self,
        p0: pt.DataFrame[P0],
        ws: pt.DataFrame[WindowedSignal],
        timestep: np.float64,
    ) -> pt.DataFrame[Bounds]:
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

        timestep = np.float64(timestep)

        # construct a container df for the bounds. needs to have the w_idx, peak_id, then bounds as [ub, lb]

        # join the window and peak dfs on time_idx, left join on peak_df

        """
        |    |   w_idx |   p_idx | param   |   p0 |
        |---:|------------:|-----------:|:--------|----------------:|
        |  0 |           1 |          0 | amp     |        42.6901  |
        |  4 |           1 |          1 | amp     |        19.9608  |
        |  8 |           1 |          2 | amp     |         2.65962 |
        | 12 |           2 |          3 | amp     |        39.8942  |
        """

        bounds_ = pd.DataFrame(
            p0.loc[:, [self.p0_sc.w_idx, self.p0_sc.p_idx, self.p0_sc.param]],
            index=p0.index,
        )
        bounds_[self.bds_sc.lb] = pd.Series([np.nan * len(p0)], dtype=pd.Float64Dtype())
        bounds_[self.bds_sc.ub] = pd.Series([np.nan * len(p0)], dtype=pd.Float64Dtype())

        bounds_ = bounds_.set_index(
            [self.bds_sc.w_idx, self.bds_sc.p_idx, self.bds_sc.param]
        )

        # amp
        amp = p0.set_index(["param"]).loc["amp"].reset_index()
        amp = (
            amp.groupby(["w_idx", "p_idx", "param"], observed=True)["p0"]
            .agg([("lb", lambda x: x * 0.1), ("ub", lambda x: x * 10)])
            .dropna()
        )
        # loc

        bounds_.loc[amp.index, [self.bds_sc.lb, self.bds_sc.ub]] = amp

        loc_b = (
            ws.loc[ws[self.ws_sc.w_type] == "peak"]
            .groupby(str(self.ws_sc.w_idx))[str(self.ws_sc.time)]
            .agg([("lb", "min"), ("ub", "max")])
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

        bounds_.loc[pd.IndexSlice[:, :, P0WIDTH], str(self.bds_sc.lb)] = timestep

        width_ub = (
            ws.loc[ws[self.ws_sc.w_type] == "peak"]
            .groupby(self.ws_sc.w_idx)[str(self.ws_sc.time)]
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

        bounds_.loc[width_ub.index, self.bds_sc.ub] = width_ub

        # skew

        bounds_.loc[pd.IndexSlice[:, :, P0SKEW], self.bds_sc.lb] = -np.inf
        bounds_.loc[pd.IndexSlice[:, :, P0SKEW], self.bds_sc.ub] = np.inf

        column_ordering = list(
            self.bds_sc.get_metadata()[str(self.bds_sc)]["columns"].keys()
        )
        column_ordering.remove("idx")

        bounds_ = bounds_.reset_index()
        bounds_ = bounds_.reindex(column_ordering, axis=1).rename_axis(
            index=self.bds_sc.idx
        )

        self.bounds = self.try_validate(bounds_, self.bds_sc)

        return self.bounds

    def _prepare_params(
        self,
        pm: DataFrame[PeakMap],
        ws: DataFrame[WindowedSignal],
        timestep: float,
    ) -> DataFrame[Params]:
        wpm = self._window_peak_map(pm, ws)

        in_p0_cols = self.get_sc_cols(self.inp0_sc)
        in_p0_cols.remove(self.inp0_sc.idx)

        in_p0_ = wpm.copy(deep=True).loc[:, in_p0_cols]
        in_p0 = self.inp0_sc.validate(in_p0_)

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


@dataclass
class PeakDeconvolver(DataPrepper):
    @pa.check_types
    def deconvolve_peaks(
        self,
        params: DataFrame[Params],
        timestep: np.float64,
    ) -> tuple[pt.DataFrame[Popt], pt.DataFrame[PSignals]]:
        params = self._prepare_params(pm, ws, timestep)

        popt_df: pd.DataFrame = self._popt_factory(windowed_signal_df, param_df)

        recon_ = self._construct_peak_signals(
            windowed_signal_df["time"].to_numpy(np.float64), popt_df
        )

        popt_df = DataFrame[Popt](popt_df)
        recon = DataFrame[PSignals](recon_)

        return popt_df, recon

    def _get_peak_report(
        self,
        popt: pt.DataFrame[Popt],
        unmixed_df: pt.DataFrame[PSignals],
        timestep: np.float64,
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
                    "retention_time": pd.Float64Dtype(),
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

        import pytest

        pytest.set_trace()
        peak_report = PReport.validate(peak_report_, lazy=True)

        return peak_report

    def _popt_factory(
        self,
        ws: DataFrame[WindowedSignal],
        params: DataFrame[Params],
        optimizer,
        fit_func,
        optimizer_kwargs={},
        verbose=True,
    ) -> pt.DataFrame[Popt]:
        popt_list = []

        import polars as pl

        ws_ = pl.from_pandas(ws)

        # windows = ws.loc[lambda df: df[self.ws_sc.w_type] == "peak"][
        #     self.ws_sc.w_idx
        # ].unique()
        params_ = pl.from_pandas(params)
        if verbose:
            windows_grpby = tqdm.tqdm(
                params_.partition_by(
                    "w_idx", maintain_order=True, as_dict=True
                ).items(),
                desc="deconvolving windows",
            )

        else:
            windows_grpby = windows

        # ws_ = ws.set_index('w_idx')

        from copy import deepcopy

        for wix, wdw in windows_grpby:
            optimizer_ = deepcopy(optimizer)
            p0 = wdw["p0"].to_numpy()
            bounds = (wdw["lb"].to_numpy(), wdw["ub"].to_numpy())
            x = ws_.filter(pl.col("w_idx") == wix)["time"].to_numpy()
            y = ws_.filter(pl.col("w_idx") == wix)["amp"].to_numpy()

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

        popt_df = pt.DataFrame[Popt](
            popt_df_.with_row_index("idx")
            .to_pandas()
            .astype({"idx": pd.Int64Dtype()})
            .set_index("idx")
        )
        return popt_df

    # @pa.check_types
    def _construct_peak_signals(
        self,
        time: FloatArray,
        popt_df: pt.DataFrame[Popt],
    ) -> pt.DataFrame[PSignals]:
        def construct_peak_signal(
            popt_df: pt.DataFrame[Popt],
            time: pt.Series,
        ) -> pd.DataFrame:
            params = popt_df.loc[
                :, ["amp", "time", "whh_half", "skew"]
            ].values.flatten()

            unmixed_signal = _compute_skewnorm(time.values, params)

            unmixed_signal_df = pd.merge(popt_df.loc[:, ["p_idx"]], time, how="cross")

            unmixed_signal_df = unmixed_signal_df.assign(amp_unmixed=unmixed_signal)

            unmixed_signal_df = unmixed_signal_df.reset_index(names="time_idx")
            
            unmixed_signal_df["time_idx"] = unmixed_signal_df["time_idx"].astype(
                pd.Int64Dtype()
            )
            unmixed_signal_df = unmixed_signal_df.loc[
                :, ["p_idx", "time_idx", "time", "amp_unmixed"]
            ]

            unmixed_signal_df = unmixed_signal_df.reset_index(drop=True)

            return unmixed_signal_df

        # remove w_idx from identifier to avoid multiindexed column

        recon_ = popt_df.groupby(
            by=["p_idx"],
            group_keys=False,
        ).apply(construct_peak_signal, *[time])

        peak_signals = recon_.reset_index(drop=True)

        return peak_signals

    def _reconstruct_signal(self, peak_signals: DataFrame[PSignals]):
        import pytest

        pytest.set_trace()

        recon_ = (
            peak_signals
            .set_index(["p_idx", "time_idx", "time"])
            .unstack(["p_idx"])
            .sum(axis=1)
            .reset_index(name="amp_recon")
            .rename_axis(index="idx")
        )

        recon = RSignal.validate(recon_, lazy=True)
        
        import pytest; pytest.set_trace()
        return recon
