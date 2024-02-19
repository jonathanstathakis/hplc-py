from typing import Self

import polars as pl
from pandera.typing import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import (
    WdwPeakMapWide,
)
from hplc_py.map_peaks.schemas import PeakMapWide

from ..map_windows.schemas import X_Windowed
from .definitions import (
    LB_KEY,
    MAXIMA_KEY,
    P0_KEY,
    PARAM_KEY,
    SKEW_KEY,
    UB_KEY,
    WHH_WIDTH_HALF_KEY,
    WHH_WIDTH_KEY,
    AMP_LB_MULT,
    AMP_UB_MULT,
    PARAM_VAL_LOC,
    PARAM_VAL_MAX,
    PARAM_VAL_SKEW,
    PARAM_VAL_WIDTH,
    SKEW_LB_SCALAR,
    SKEW_UB_SCALAR,
)
from .schemas import P0, Bounds, InP0, Params
from .typing import p0_param_cats


class DataPrepper:
    """
    Prepares the data for input into curve_fit.optimize.
    """

    def __init__(self):

        self._p0_param_cats = p0_param_cats

        self._p0_key = P0_KEY
        self._params_key = PARAM_KEY
        self._lb_key = LB_KEY
        self._ub_key = UB_KEY
        self._skew_key = SKEW_KEY
        self._whh_half_key = WHH_WIDTH_HALF_KEY
        self._maxima_key = MAXIMA_KEY
        self._whh_key = WHH_WIDTH_KEY

    def fit(
        self,
        pm: DataFrame[PeakMapWide],
        X_w: DataFrame[X_Windowed],
        X_key: str,
        X_idx_key: str,
        w_idx_key: str,
        w_type_key: str,
        p_idx_key: str,
        whh_width_key: str,
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
        self._whh_key = whh_width_key

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
            whh_width_half_key=self._whh_half_key,
            maxima_key=self._maxima_key,
            whh_width_key=self._whh_key,
            amp_lb_mult=AMP_LB_MULT,
            amp_ub_mult=AMP_UB_MULT,
            param_val_loc=PARAM_VAL_LOC,
            param_val_maxima=PARAM_VAL_MAX,
            param_val_skew=PARAM_VAL_SKEW,
            param_val_width=PARAM_VAL_WIDTH,
            skew_lb_scalar=SKEW_LB_SCALAR,
            skew_ub_scalar=SKEW_UB_SCALAR,
        )

        return self


def params_factory(
    pm: DataFrame[PeakMapWide],
    X_w: DataFrame[X_Windowed],
    timestep: float,
    X_key: str,
    time_key: str,
    whh_key: str,
    p0_param_cats: pl.Enum,
    whh_width_key: str,
    X_idx_key: str,
    w_idx_key: str,
    w_type_key: str,
    p_idx_key: str,
    param_key: str,
    p0_key: str,
    lb_key: str,
    ub_key: str,
    whh_width_half_key: str,
    maxima_key: str,
    skew_key: str,
    amp_ub_mult: float,
    amp_lb_mult: float,
    skew_lb_scalar: float,
    skew_ub_scalar: float,
    param_val_maxima: str,
    param_val_loc: str,
    param_val_width: str,
    param_val_skew: str,
):
    """
    Prepare the parameter input to `optimize`, i.e. the lb, p0 and ub for each parameter
    of the skewnorm model.

    :param pm: peakmap table
    :type pm: DataFrame[PeakMapWide]
    :param ws: windowed signal table
    :type ws: DataFrame[X_Windowed]
    :param timestep: the timestep
    :type timestep: float
    :return: the parameter table in long form with the 4 parameters of each peak of
    each window
    :rtype: DataFrame[Params]
    """

    wpm = window_peak_map(
        peak_map=pm,
        X_w=X_w,
        t_idx_key=X_idx_key,
        w_idx_key=w_idx_key,
        w_type_key=w_type_key,
        X_idx_key=X_idx_key,
    )

    # the input to `p0_factory` is depicted in the In_p0 schema. Use it to subset
    # the wpm then submit to to `p0_factory`

    p0 = p0_factory(
        wpm=wpm,
        maxima_key=maxima_key,
        X_idx_key=X_idx_key,
        p0_key=p0_key,
        p_idx_key=p_idx_key,
        param_key=param_key,
        timestep=timestep,
        w_idx_key=w_idx_key,
        whh_width_key=whh_width_key,
        skew_key=skew_key,
        whh_width_half_key=whh_width_half_key,
        p0_param_cat_dtype=p0_param_cats,
        param_val_loc=param_val_loc,
        param_val_maxima=param_val_maxima,
        param_val_skew=param_val_skew,
        param_val_width=param_val_width,
    )

    bounds = bounds_factory(
        p0=p0,
        X_w=X_w,
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
        whh_width_half_key=whh_width_half_key,
        skew_key=skew_key,
        amp_ub_mult=amp_ub_mult,
        amp_lb_mult=amp_lb_mult,
        maxima_key=maxima_key,
        param_val_loc=param_val_loc,
        param_val_maxima=param_val_maxima,
        param_val_skew=param_val_skew,
        param_val_width=param_val_width,
        skew_lb_scalar=skew_lb_scalar,
        skew_ub_scalar=skew_ub_scalar,
        param_cats=p0_param_cats,
    )

    # join the p0 and bounds tables

    join_cols = [w_idx_key, p_idx_key, param_key]
    params: DataFrame[Params] = (
        p0
        .pipe(pl.from_pandas)
        .join(
            bounds
            .pipe(pl.from_pandas),
            on=join_cols,
            how="left",
        )
        .to_pandas()
        .pipe(Params.validate, lazy=True)
        .pipe(DataFrame[Params])
    )  # fmt: skip

    return params


def bounds_factory(
    p0: DataFrame[P0],
    X_w: DataFrame[X_Windowed],
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
    whh_width_half_key: str,
    maxima_key: str,
    skew_key: str,
    amp_ub_mult: float,
    amp_lb_mult: float,
    skew_lb_scalar: float,
    skew_ub_scalar: float,
    param_val_maxima: str,
    param_val_loc: str,
    param_val_width: str,
    param_val_skew: str,
    param_cats: pl.Enum,
) -> DataFrame[Bounds]:
    """
    Build a default bounds df from the `signal_df`, `peak_df`, and `window_df`, intended for joining with the p0 table. Structure is depicted in `.schemas.Bounds`

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

    # The Skewnorm Distribution

    The following is a description of the parameters of the skewnorm distribution.

    ## Skew

    skew is determined by alpha, and right skewed if alpha > 0, left skewed if alpha < 0.
    """

    idx_cols = [w_idx_key, p_idx_key]

    window_bounds = find_window_bounds(
        X_w=X_w, w_idx_key=w_idx_key, lb_key=lb_key, ub_key=ub_key, X_idx_key=X_idx_key
    )

    bounds_maxima = set_bounds_maxima(
        p0=p0,
        idx_cols=idx_cols,
        maxima_key=maxima_key,
        lb_key=lb_key,
        ub_key=ub_key,
        amp_lb_mult=amp_lb_mult,
        amp_ub_mult=amp_ub_mult,
        p0_key=p0_key,
        param_key=param_key,
        param_val=param_val_maxima,
    )

    bounds_loc = set_bounds_loc(
        window_bounds=window_bounds,
        p0=p0,
        idx_cols=idx_cols,
        lb_key=lb_key,
        ub_key=ub_key,
        param_val=param_val_loc,
        param_key=param_key,
        p0_key=p0_key,
    )

    bounds_width = set_bounds_width(
        p0=p0,
        window_bounds=window_bounds,
        timestep=timestep,
        idx_cols=idx_cols,
        lb_key=lb_key,
        ub_key=ub_key,
        param_key=param_key,
        param_val=param_val_width,
        p_idx_key=p_idx_key,
        w_idx_key=w_idx_key,
    )

    bounds_skew = set_bounds_skew(
        p0=p0,
        p_idx_key=p_idx_key,
        w_idx_key=w_idx_key,
        idx_cols=idx_cols,
        lb_key=lb_key,
        ub_key=ub_key,
        skew_lb_scalar=skew_lb_scalar,
        skew_ub_scalar=skew_ub_scalar,
        param_val=param_val_skew,
        param_key=param_key,
    )
    bounds = (
        pl.concat(
            [
                df.with_columns(pl.col([lb_key, ub_key]).cast(float))
                for df in [bounds_maxima, bounds_loc, bounds_width, bounds_skew]
            ]
        )
        .with_columns(pl.col(param_key).cast(param_cats))
        .sort([w_idx_key, p_idx_key, param_key])
        .to_pandas()
        .pipe(Bounds.validate, lazy=True)
        .pipe(DataFrame[Bounds])
    )

    return bounds


def find_window_bounds(
    X_w: DataFrame[X_Windowed],
    w_idx_key: str,
    X_idx_key: str,
    lb_key: str,
    ub_key: str,
) -> pl.DataFrame:
    """
    Find the lower and upper loc bounds for each window.

    Note: this behavior is the same as map_windows.get_window_X_idx_bounds, but that function currently includes the p_idx in the input schema. As it is not a complicated calculation I will essentially recreate it here until a commonality can be determined
    """

    bounds_loc = (
        X_w.pipe(pl.from_pandas)
        .group_by([w_idx_key])
        .agg(
            pl.col(X_idx_key).min().alias(lb_key),
            pl.col(X_idx_key).max().alias(ub_key),
        )
    )

    return bounds_loc


def set_bounds_skew(
    p0: DataFrame[P0],
    w_idx_key: str,
    p_idx_key: str,
    idx_cols: list[str],
    ub_key: str,
    lb_key: str,
    skew_lb_scalar: float,
    skew_ub_scalar: float,
    param_key: str,
    param_val: str,
) -> pl.DataFrame:

    bounds_skew = (
        p0
        .pipe(pl.from_pandas)
        .select(pl.col([w_idx_key,p_idx_key]))
        .unique()
        .with_columns(
            pl.lit(param_val)
            .alias(param_key)
        )
        .select(
            pl.col(idx_cols),
            pl.col(param_key),
            pl.lit(skew_lb_scalar)
            .alias(lb_key),
            pl.lit(skew_ub_scalar)
            .alias(ub_key),
    )
        )  # fmt: skip
    return bounds_skew


def set_bounds_width(
    p0: DataFrame[P0],
    window_bounds: pl.DataFrame,
    timestep: float,
    ub_key: str,
    lb_key: str,
    idx_cols: list[str],
    param_key: str,
    param_val: str,
    p_idx_key: str,
    w_idx_key: str,
) -> pl.DataFrame:
    """
    set the bounds on the half of the width measured at half height. The bounds are defined as the timestep for the lower bound and half the length of the window for the upper.

    """

    width_bounds: pl.DataFrame = window_bounds.with_columns(
        pl.lit(timestep).alias(lb_key)
    ).select(pl.col(w_idx_key), pl.col(lb_key), pl.col(ub_key).truediv(2).alias(ub_key))

    bounds_width: pl.DataFrame = (
        p0
            .pipe(pl.from_pandas)
            .select([w_idx_key,p_idx_key])
            .unique()
            .with_columns(
                pl.lit(param_val)
                .alias(param_key)
                          )
            .join(
                width_bounds,
                on=w_idx_key,
                how='left',
                # validate="m:1",
            
        )
    )  # fmt: skip

    return bounds_width


def set_bounds_loc(
    idx_cols: list[str],
    p0: DataFrame[P0],
    window_bounds: pl.DataFrame,
    lb_key: str,
    ub_key: str,
    param_val: str,
    param_key: str,
    p0_key: str,
):
    """
        location (time or otherwise) bounds are defined as the minimum and maximum time of each window.
    ZA
        This is achieved by taking the pre-calculated window bounds and joining to each p_idx via their associated w_idx.
    """

    bounds_loc: pl.DataFrame = (
        p0.pipe(pl.from_pandas)
        .pivot(index=idx_cols, columns=param_key, values=p0_key)
        .select(pl.col(idx_cols))
        .with_columns(pl.lit(param_val).alias(param_key))
        .join(window_bounds, how="left", on="w_idx", validate="m:1")
    )

    return bounds_loc


def set_bounds_maxima(
    p0: DataFrame[P0],
    idx_cols: list[str],
    maxima_key: str,
    amp_lb_mult: float,
    amp_ub_mult: float,
    lb_key: str,
    ub_key: str,
    param_key: str,
    p0_key: str,
    param_val: str,
) -> pl.DataFrame:
    """
    The amplitude bounds are defined as a multiple up and down of the maxima
    """

    bounds_maxima = (
            pl.from_pandas(p0)
            .pivot(index=idx_cols, columns=param_key, values=p0_key)
            .select(
                pl.col(idx_cols),
                pl.lit(param_val).alias(param_key),
                pl.col(maxima_key).mul(amp_lb_mult).alias(lb_key),
                pl.col(maxima_key).mul(amp_ub_mult).alias(ub_key)
                )
        )  # fmt: skip

    return bounds_maxima


def window_peak_map(
    peak_map: DataFrame[PeakMapWide],
    X_w: DataFrame[X_Windowed],
    t_idx_key: str,
    X_idx_key: str,
    w_type_key: str,
    w_idx_key: str,
) -> DataFrame[WdwPeakMapWide]:
    """
    add w_idx to to peak map for later lookups
    """

    wpm: pl.DataFrame = (
        peak_map
        .pipe(pl.from_pandas)
        .rename({t_idx_key: X_idx_key})
        .join(
            pl.from_pandas(X_w)
            .select([w_type_key, w_idx_key, X_idx_key]),
            how="left",
            on=X_idx_key,
            validate="1:1",
        )
        .select(pl.col([w_type_key, w_idx_key]), pl.exclude([w_type_key, w_idx_key]))
    )  # fmt: skip

    if wpm.select(pl.col(w_type_key).is_in(["interpeak"]).any()).item():
        raise ValueError("peak has been assigned to interpeak region.")

    return DataFrame[WdwPeakMapWide](wpm.to_pandas())


def p0_factory(
    wpm: DataFrame[WdwPeakMapWide],
    timestep: float,
    maxima_key: str,
    X_idx_key: str,
    whh_width_key: str,
    w_idx_key: str,
    p_idx_key: str,
    param_key: str,
    p0_key: str,
    skew_key: str,
    whh_width_half_key: str,
    p0_param_cat_dtype: pl.Enum,
    param_val_maxima: str,
    param_val_width: str,
    param_val_loc: str,
    param_val_skew: str,
) -> DataFrame[P0]:
    """
    Build a table of initial guesses for each peak in each window.
    """
    # window the peak map
    # assign skew as zero as per definition
    # assign whh as half peak whh as per definition, in time units
    in_p0_keys = [w_idx_key, p_idx_key, maxima_key, X_idx_key, whh_width_key]

    p0: DataFrame[P0] = (
        wpm
            .loc[:, in_p0_keys]
            .pipe(DataFrame[InP0])
            .pipe(pl.from_pandas) #type: ignore
            .select(
                pl.col([w_idx_key, p_idx_key]),
                pl.col(maxima_key).alias(param_val_maxima),
                pl.col(X_idx_key).alias(param_val_loc),
                pl.col(whh_width_key).truediv(2).alias(param_val_width),
                pl.lit(0).cast(float).alias(param_val_skew),
                          )
            .melt(id_vars=[w_idx_key,p_idx_key], variable_name=param_key, value_name=p0_key)
            .with_columns(pl.col(param_key).cast(p0_param_cats))
            .sort([w_idx_key, p_idx_key, param_key])
            .to_pandas()
            .pipe(P0.validate, lazy=True)
            )  # fmt: skip

    return p0
