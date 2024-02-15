from typing import Self
import numpy as np
import pandas as pd
import polars as pl
from numpy import float64
from pandera.typing import DataFrame

from hplc_py import P0AMP, P0SKEW, P0TIME, P0WIDTH
from hplc_py.hplc_py_typing.hplc_py_typing import (
    P0,
    Bounds,
    InP0,
    Params,
    WdwPeakMapWide,
    X_Windowed,
)
from hplc_py.map_peaks.map_peaks import PeakMapWide


class DataPrepper:
    """
    Prepares the data for input into curve_fit.optimize.
    """

    def __init__(self):

        self._p0_param_cats = pd.CategoricalDtype(
            [
                P0AMP,
                P0TIME,
                P0WIDTH,
                P0SKEW,
            ],
            ordered=True,
        )

        self._p0_key = "p0"
        self._params_key = "params"
        self._lb_key = "lb"
        self._ub_key = "ub"
        self._skew_key = "skew"
        self._whh_half_key = "whh_half"

    def fit(
        self,
        pm: DataFrame[PeakMapWide],
        X_w: DataFrame[X_Windowed],
        X_key: str,
        X_idx_key: str,
        w_idx_key: str,
        w_type_key: str,
        p_idx_key: str,
        whh_key: str,
        time_key: str,
        timestep: float,
    ) -> Self:
        self._pm = pm
        self._X_w = X_w
        self._timestep = timestep
        self._w_idx_key = w_idx_key
        self._w_type_key = w_type_key
        self._p_idx_key = p_idx_key
        self._X_idx_key = X_idx_key
        self._X_key = X_key
        self._time_key = time_key
        self._whh_key = whh_key

        return self

    def transform(
        self,
    ) -> Self:

        self.params = params_factory(
            pm=self._pm,
            X_w=self._X_w,
            timestep=self._timestep,
            whh_key=self._whh_key,
            w_idx_key=self._w_idx_key,
            time_key=self._time_key,
            p_idx_key=self._p_idx_key,
            X_key=self._X_key,
            X_idx_key=self._X_idx_key,
            p0_key=self._p0_key,
            param_key=self._params_key,
            w_type_key=self._w_type_key,
            lb_key=self._lb_key,
            p0_param_cats=self._p0_param_cats,
            skew_key=self._skew_key,
            ub_key=self._ub_key,
            whh_half_key=self._whh_half_key,
        )

        return self


def params_factory(
    pm: DataFrame[PeakMapWide],
    X_w: DataFrame[X_Windowed],
    timestep: float,
    X_key: str,
    time_key: str,
    X_idx_key: str,
    p_idx_key: str,
    w_idx_key: str,
    w_type_key: str,
    whh_key: str,
    p0_key: str,
    param_key: str,
    whh_half_key: str,
    p0_param_cats: pd.CategoricalDtype,
    skew_key: str,
    lb_key: str,
    ub_key: str,
):
    """
    Prepare the parameter input to `optimize`, i.e. the lb, p0 and ub for each parameter
    of the skewnorm model.

    :param pm: peakmap table
    :type pm: DataFrame[PeakMapWide]
    :param ws: windowed signal table
    :type ws: DataFrame[X_Windowed]
    :param timestep: the timestep
    :type timestep: float64
    :return: the parameter table in long form with the 4 parameters of each peak of
    each window
    :rtype: DataFrame[Params]
    """

    wpm = window_peak_map(
        pm=pm,
        X_w=X_w,
        t_idx_key=X_idx_key,
        w_idx_key=w_idx_key,
        w_type_key=w_type_key,
        X_idx_key=X_idx_key,
    )

    # the input to `p0_factory` is depicted in the In_p0 schema. Use it to subset
    # the wpm then submit to to `p0_factory`

    in_p0_ = wpm.loc[:, [w_idx_key, p_idx_key, X_key, time_key, whh_key]]

    InP0.validate(in_p0_, lazy=True)

    in_p0 = DataFrame[InP0](in_p0_)

    p0 = p0_factory(
        wpm=in_p0,
        p0_key=p0_key,
        p_idx_key=p_idx_key,
        param_key=param_key,
        timestep=timestep,
        w_idx_key=w_idx_key,
        whh_key=whh_key,
        skew_key=skew_key,
        whh_half_key=whh_half_key,
        p0_param_cats=p0_param_cats,
    )

    bounds = bounds_factory(
        p0=p0,
        ws=X_w,
        timestep=timestep,
        X_key=X_key,
        X_idx_key=X_idx_key,
        w_idx_key=w_idx_key,
        w_type_key=w_type_key,
        p_idx_key=p_idx_key,
        param_key=param_key,
        p0_key=p0_key,
        lb_key=lb_key,
        ub_key=ub_key,
        time_key=time_key,
    )

    # join the p0 and bounds tables

    join_cols = [w_idx_key, p_idx_key, param_key]

    p0_ = p0.reset_index().set_index(join_cols)
    bounds_ = bounds.set_index(join_cols)

    params = p0_.join(bounds_, how="left", validate="1:1").reset_index()

    Params.validate(params, lazy=True)
    return DataFrame[Params](params)


def bounds_factory(
    p0: DataFrame[P0],
    ws: DataFrame[X_Windowed],
    timestep: float,
    X_key: str,
    X_idx_key: str,
    w_idx_key: str,
    w_type_key: str,
    p_idx_key: str,
    param_key: str,
    p0_key: str,
    lb_key: str,
    ub_key: str,
    time_key: str,
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

    bound_cols = [w_idx_key, p_idx_key, param_key]
    bounds = pd.DataFrame(p0.loc[:, bound_cols], index=p0.index)

    bounds[lb_key] = pd.Series([np.nan * len(p0)], dtype=float64)
    bounds[ub_key] = pd.Series([np.nan * len(p0)], dtype=float64)

    bounds = bounds.set_index([w_idx_key, p_idx_key, param_key])

    # amp
    amp = p0.set_index([param_key]).loc[X_key].reset_index()
    amp = (
        amp.groupby([w_idx_key, p_idx_key, param_key], observed=True)[p0_key]
        .agg([(lb_key, lambda x: x * 0.1), (ub_key, lambda x: x * 10)])  # type: ignore
        .dropna()
    )
    # loc

    bounds.loc[amp.index, [lb_key, ub_key]] = amp

    loc_b = (
        ws.loc[ws[w_type_key] == "peak"]
        .groupby(str(w_idx_key))[time_key]
        .agg([(lb_key, "min"), (ub_key, "max")])  # type: ignore
        .reset_index()
        .assign(param=P0TIME)
        .set_index([str(w_idx_key), param_key])
    )

    # get the peak idx from p0
    loc_b = (
        p0.set_index([w_idx_key, param_key])
        .drop(p0_key, axis=1)
        .join(loc_b.reset_index().set_index([w_idx_key, param_key]), how="right")
        .reset_index()
    )

    loc_b = loc_b.set_index([w_idx_key, p_idx_key, param_key])
    bounds.loc[loc_b.index, [lb_key, ub_key]] = loc_b

    bounds.loc[pd.IndexSlice[:, :, P0WIDTH], lb_key] = timestep  # type: ignore

    width_ub = (
        ws.loc[ws[w_type_key] == "peak"]
        .groupby(w_idx_key)[time_key]
        .agg(lambda x: (x.max() - x.min()) / 2)
        .rename(ub_key)
        .reset_index()
        .assign(param=P0WIDTH)
        .set_index([w_idx_key, param_key])
    )

    width_ub = (
        p0.set_index([w_idx_key, param_key])
        .drop(p0_key, axis=1)
        .join(width_ub.reset_index().set_index([w_idx_key, param_key]), how="right")
        .reset_index()
        .set_index([w_idx_key, p_idx_key, param_key])
    )

    bounds.loc[width_ub.index, ub_key] = width_ub

    # skew

    bounds.loc[pd.IndexSlice[:, :, P0SKEW], lb_key] = -np.inf
    bounds.loc[pd.IndexSlice[:, :, P0SKEW], ub_key] = np.inf

    column_ordering = [w_idx_key, p_idx_key, param_key, lb_key, ub_key]
    bounds = bounds.reset_index.reindex(column_ordering, axis=1)

    Bounds.validate(bounds, lazy=True)

    return DataFrame[Bounds](bounds)


def window_peak_map(
    pm: DataFrame[PeakMapWide],
    X_w: DataFrame[X_Windowed],
    t_idx_key: str,
    X_idx_key: str,
    w_type_key: str,
    w_idx_key: str,
) -> DataFrame[WdwPeakMapWide]:
    """
    add w_idx to to peak map for later lookups
    """

    X_w_pl: pl.DataFrame = pl.from_pandas(X_w)
    pm_pl: pl.DataFrame = pl.from_pandas(pm)

    breakpoint()
    wpm: pl.DataFrame = (
        pm_pl.rename({t_idx_key: X_idx_key})
        .join(
            X_w_pl.select([w_type_key, w_idx_key, X_idx_key]),
            how="left",
            on=X_idx_key,
            validate="1:1",
        )
        .pipe(lambda df: df if breakpoint() else df)
        .select(pl.col([w_type_key, w_idx_key]), pl.exclude([w_type_key, w_idx_key]))
    )

    if wpm.select(pl.col(w_type_key).is_in(["interpeak"]).any()).item():
        raise ValueError("peak has been assigned to interpeak region.")

    breakpoint()

    wpm_pd = wpm.to_pandas()
    WdwPeakMapWide.validate(wpm_pd, lazy=True)

    return WdwPeakMapWide[DataFrame](wpm_pd)  # false error?


def p0_factory(
    wpm: DataFrame[InP0],
    timestep: float,
    whh_key: str,
    w_idx_key: str,
    p_idx_key: str,
    param_key: str,
    p0_key: str,
    skew_key: str,
    whh_half_key: str,
    p0_param_cats: pd.CategoricalDtype,
) -> DataFrame[P0]:
    """
    Build a table of initial guesses for each peak in each window.
    """
    # window the peak map
    # assign skew as zero as per definition
    # assign whh as half peak whh as per definition, in time units

    p0_ = wpm.copy(deep=True)
    p0_[whh_half_key] = pd.Series([0.0] * len(p0_), dtype=float64)
    p0_[skew_key] = p0_.pop(whh_key).div(2).mul(float(timestep))

    # set index as idx, w_idx, p_idx
    p0_ = p0_.set_index([w_idx_key, p_idx_key], append=True)

    # go from wide to long with index as above + a param col, 1 value col p0
    p0_ = (
        p0_.stack()
        .reset_index(level=3)
        .rename(
            {"level_3": param_key, 0: p0_key},
            axis=1,
            errors="raise",
        )
    )

    # set the param col as an ordered categorical
    p0_[param_key] = pd.Categorical(p0_[param_key], dtype=p0_param_cats)

    # add the param col to index and sort
    p0_ = p0_.set_index(param_key, append=True).sort_index()

    # return index cols to columns
    p0 = p0_.reset_index([w_idx_key, p_idx_key, param_key]).reset_index(drop=True)

    # reset to range index

    P0.validate(p0, lazy=True)

    return DataFrame[P0](p0)
