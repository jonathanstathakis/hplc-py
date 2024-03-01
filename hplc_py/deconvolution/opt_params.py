from typing import Self

import polars as pl
from pandera.typing import DataFrame
from hplc_py.map_windows import schemas as mw_schs
from hplc_py.deconvolution import definitions as dc_defs
from hplc_py.deconvolution import schemas as dc_schs


class DataPrepper:
    """
    Prepares the data for input into curve_fit.optimize.
    """

    def fit(
        self,
        peak_msnts_windowed: DataFrame[dc_schs.PeakMsntsWindowed],
        X_w: DataFrame[mw_schs.X_Windowed],
        timestep: float,
    ) -> Self:
        self.opt_params_peak_input = peak_msnts_windowed
        self._X_w = X_w
        self._timestep = timestep

        return self

    def transform(
        self,
    ) -> Self:

        self.params = params_factory(
            peak_msnts_windowed=self.opt_params_peak_input,
            X_w=self._X_w,
            timestep=self._timestep,
        )

        return self


def params_factory(
    peak_msnts_windowed,  #: DataFrame[dc_schs.PeakMsntsWindowed],
    X_w: DataFrame[mw_schs.X_Windowed],
    timestep: float,
    amp_lb_mult: float = dc_defs.VAL_AMP_LB_MULT,
    amp_ub_mult: float = dc_defs.VAL_AMP_UP_MULT,
    skew_lb_scalar: float = dc_defs.VAL_SKEW_LB_SCALAR,
    skew_ub_scalar: float = dc_defs.VAL_SKEW_UB_SCALAR,
    param_val_maxima: str = dc_defs.MAXIMA,
    param_val_loc: str = dc_defs.LOC,
    param_val_width: str = dc_defs.WIDTH,
    param_val_skew: str = dc_defs.SKEW,
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
    :rtype: DataFrame[dc_schs.Params]
    """

    p0: DataFrame[dc_schs.P0] = p0_factory(
        peak_msnts_windowed=peak_msnts_windowed,
    )
    breakpoint()
    bounds: DataFrame[dc_schs.Bounds] = bounds_factory(
        p0=p0,
        X_w=X_w,
        timestep=timestep,
        amp_ub_mult=amp_ub_mult,
        amp_lb_mult=amp_lb_mult,
        param_val_loc=param_val_loc,
        param_val_maxima=param_val_maxima,
        param_val_skew=param_val_skew,
        param_val_width=param_val_width,
        skew_lb_scalar=skew_lb_scalar,
        skew_ub_scalar=skew_ub_scalar,
    )

    # join the p0 and bounds tables

    join_cols = [dc_defs.W_IDX, dc_defs.P_IDX, dc_defs.PARAM]
    params: DataFrame[dc_schs.Params] = (
        p0
        .pipe(pl.from_pandas)
        .join(
            bounds
            .pipe(pl.from_pandas),
            on=join_cols,
            how="left",
        )
        .to_pandas()
        .pipe(dc_schs.Params.validate, lazy=True)
        .pipe(DataFrame[dc_schs.Params])
    )  # fmt: skip

    return params


def bounds_factory(
    p0: DataFrame[dc_schs.P0],
    X_w: DataFrame[dc_schs.X_Windowed],
    timestep: float,
    amp_ub_mult: float,
    amp_lb_mult: float,
    skew_lb_scalar: float,
    skew_ub_scalar: float,
    param_val_maxima: str,
    param_val_loc: str,
    param_val_width: str,
    param_val_skew: str,
) -> DataFrame[dc_schs.Bounds]:
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

    TODO: remove local references to global constants. i.e. the "VAL" class of kwargs.
    """

    idx_cols = [dc_defs.W_IDX, dc_defs.P_IDX]

    window_bounds = find_window_bounds(X_w=X_w)

    bounds_maxima = set_bounds_maxima(
        p0=p0,
        idx_cols=idx_cols,
        amp_lb_mult=amp_lb_mult,
        amp_ub_mult=amp_ub_mult,
        param_val=param_val_maxima,
    )

    bounds_loc = set_bounds_loc(
        window_bounds=window_bounds,
        p0=p0,
        idx_cols=idx_cols,
        param_val=param_val_loc,
    )

    bounds_width = set_bounds_width(
        p0=p0,
        window_bounds=window_bounds,
        timestep=timestep,
        idx_cols=idx_cols,
        param_val=param_val_width,
    )

    bounds_skew = set_bounds_skew(
        p0=p0,
        idx_cols=idx_cols,
        skew_lb_scalar=skew_lb_scalar,
        skew_ub_scalar=skew_ub_scalar,
        param_val=param_val_skew,
    )
    bounds = (
        pl.concat(
            [
                df.with_columns(pl.col([dc_defs.KEY_LB, dc_defs.KEY_UB]).cast(float))
                for df in [bounds_maxima, bounds_loc, bounds_width, bounds_skew]
            ]
        )
        .with_columns(pl.col(dc_defs.PARAM).cast(dc_defs.p0_param_cats))
        .sort([dc_defs.W_IDX, dc_defs.P_IDX, dc_defs.PARAM])
        .to_pandas()
        .pipe(dc_schs.Bounds.validate, lazy=True)
        .pipe(DataFrame[dc_schs.Bounds])
    )

    return bounds


def find_window_bounds(
    X_w: DataFrame[mw_schs.X_Windowed],
) -> pl.DataFrame:
    """
    Find the lower and upper loc bounds for each window.

    Note: this behavior is the same as map_windows.get_window_X_idx_bounds, but that function currently includes the p_idx in the input schema. As it is not a complicated calculation I will essentially recreate it here until a commonality can be determined
    """

    bounds_loc = (
        X_w.pipe(pl.from_pandas)
        .group_by([dc_defs.W_IDX])
        .agg(
            pl.col(dc_defs.X_IDX).min().alias(dc_defs.KEY_LB),
            pl.col(dc_defs.X_IDX).max().alias(dc_defs.KEY_UB),
        )
    )

    return bounds_loc


def set_bounds_skew(
    p0: DataFrame[dc_schs.P0],
    idx_cols: list[str],
    skew_lb_scalar: float,
    skew_ub_scalar: float,
    param_val: str,
) -> pl.DataFrame:

    bounds_skew = (
        p0
        .pipe(pl.from_pandas)
        .select(pl.col([dc_defs.W_IDX,dc_defs.P_IDX]))
        .unique()
        .with_columns(
            pl.lit(param_val)
            .alias(dc_defs.PARAM)
        )
        .select(
            pl.col(idx_cols),
            pl.col(dc_defs.PARAM),
            pl.lit(skew_lb_scalar)
            .alias(dc_defs.KEY_LB),
            pl.lit(skew_ub_scalar)
            .alias(dc_defs.KEY_UB),
    )
        )  # fmt: skip
    return bounds_skew


def set_bounds_width(
    p0: DataFrame[dc_schs.P0],
    window_bounds: pl.DataFrame,
    timestep: float,
    idx_cols: list[str],
    param_val: str,
) -> pl.DataFrame:
    """
    set the bounds on the half of the width measured at half height. The bounds are defined as the timestep for the lower bound and half the length of the window for the upper.

    """

    width_bounds: pl.DataFrame = window_bounds.with_columns(
        pl.lit(timestep).alias(dc_defs.KEY_LB)
    ).select(
        pl.col(dc_defs.W_IDX),
        pl.col(dc_defs.KEY_LB),
        pl.col(dc_defs.KEY_UB).truediv(2).alias(dc_defs.KEY_UB),
    )

    bounds_width: pl.DataFrame = (
        p0
            .pipe(pl.from_pandas)
            .select([dc_defs.W_IDX,dc_defs.P_IDX])
            .unique()
            .with_columns(
                pl.lit(param_val)
                .alias(dc_defs.PARAM)
                          )
            .join(
                width_bounds,
                on=dc_defs.W_IDX,
                how='left',
                # validate="m:1",
            
        )
    )  # fmt: skip

    return bounds_width


def set_bounds_loc(
    idx_cols: list[str],
    p0: DataFrame[dc_schs.P0],
    window_bounds: pl.DataFrame,
    param_val: str,
):
    """
        location (time or otherwise) bounds are defined as the minimum and maximum time of each window.
    ZA
        This is achieved by taking the pre-calculated window bounds and joining to each p_idx via their associated w_idx.
    """

    bounds_loc: pl.DataFrame = (
        p0.pipe(pl.from_pandas)
        .pivot(index=idx_cols, columns=dc_defs.PARAM, values=dc_defs.KEY_P0)
        .select(pl.col(idx_cols))
        .with_columns(pl.lit(param_val).alias(dc_defs.PARAM))
        .join(window_bounds, how="left", on="w_idx", validate="m:1")
    )

    return bounds_loc


def set_bounds_maxima(
    p0: DataFrame[dc_schs.P0],
    idx_cols: list[str],
    amp_lb_mult: float,
    amp_ub_mult: float,
    param_val: str,
) -> pl.DataFrame:
    """
    The amplitude bounds are defined as a multiple up and down of the maxima
    """

    bounds_maxima = (
            pl.from_pandas(p0)
            .pivot(index=idx_cols, columns=dc_defs.PARAM, values=dc_defs.KEY_P0)
            .select(
                pl.col(idx_cols),
                pl.lit(param_val).alias(dc_defs.PARAM),
                pl.col(dc_defs.MAXIMA).mul(amp_lb_mult).alias(dc_defs.KEY_LB),
                pl.col(dc_defs.MAXIMA).mul(amp_ub_mult).alias(dc_defs.KEY_UB)
                )
        )  # fmt: skip

    return bounds_maxima


def p0_factory(
    peak_msnts_windowed,  #: DataFrame[dc_schs.PeakMsntsWindowed],
) -> DataFrame[dc_schs.P0]:
    """
    Build a table of initial guesses for each peak in each window using the peak maxima, the peak width at half height, peak time location (X_idx), and default skews of zero.
    """
    # window the peak map
    # assign skew as zero as per definition
    # assign whh as half peak whh as per definition, in time units
    breakpoint()
    # todo: manually add dim, unit columns such that dim is x or y, unit is for example idx, time, amp etc.
    
    p0: DataFrame[dc_schs.P0] = (
        peak_msnts_windowed
            .pipe(pl.from_pandas) #type: ignore
            .pivot(index=['w_type','w_idx','p_idx'], columns=['dim'], values='value')
            .select(
                pl.col([dc_defs.W_TYPE, dc_defs.W_IDX, dc_defs.P_IDX]),
                pl.col(dc_defs.X).alias(dc_defs.MAXIMA),
                pl.col(dc_defs.X_IDX).alias(dc_defs.LOC),
                pl.col(dc_defs.WIDTH).truediv(2).alias(dc_defs.WIDTH),
                pl.lit(0).cast(float).alias(dc_defs.SKEW),
                          )
            .melt(id_vars=[dc_defs.W_TYPE, dc_defs.W_IDX,dc_defs.P_IDX],variable_name=dc_defs.PARAM, value_name=dc_defs.KEY_P0)
            .with_columns(pl.col(dc_defs.PARAM).cast(dc_defs.p0_param_cats))
            .sort([dc_defs.W_IDX, dc_defs.P_IDX, dc_defs.PARAM])
            .to_pandas()
            .pipe(dc_schs.P0.validate, lazy=True)
            )  # fmt: skip

    return p0
