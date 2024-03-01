import pandas as pd
import pytest
from pandera.typing import DataFrame

from hplc_py.common.definitions import (
    AMP_LB_MULT,
    AMP_UB_MULT,
    LB_KEY,
    MAXIMA_KEY,
    P0_KEY,
    P_IDX_KEY,
    PARAM_KEY,
    SKEW_KEY,
    UB_KEY,
    W_IDX_KEY,
    W_TYPE_KEY,
    WHH_WIDTH_HALF_KEY,
    WHH_WIDTH_KEY,
    X_IDX,
    X,
    SKEW_LB_SCALAR,
    SKEW_UB_SCALAR,
    PARAM_VAL_MAX,
    PARAM_VAL_LOC,
    PARAM_VAL_WIDTH,
    PARAM_VAL_SKEW,
)
from hplc_py.deconvolution.opt_params import (
    DataPrepper,
    bounds_factory,
    p0_factory,
)
from hplc_py.deconvolution.schemas import P0, Bounds, Params
from hplc_py.deconvolution.definitions import p0_param_cats
from hplc_py.hplc_py_typing.hplc_py_typing import (
    WdwPeakMapWide,
)

from hplc_py.map_peaks.schemas import PeakMap
from hplc_py.map_windows.schemas import X_Windowed
from hplc_py.pipeline.deconvolution.deconv import window_peak_map


@pytest.fixture
def p0_param_cats_fix():
    return p0_param_cats


@pytest.fixture
def dp():
    dp = DataPrepper()
    return dp


@pytest.fixture
def wpm(
    peak_map: DataFrame[PeakMap],
    X_windowed,
) -> DataFrame[WdwPeakMapWide]:
    wpm: DataFrame[WdwPeakMapWide] = window_peak_map(
        peak_msnts=peak_map,
        X_w=X_windowed,
        t_idx_key=X_IDX,
        w_idx_key=W_IDX_KEY,
        w_type_key=W_TYPE_KEY,
        X_idx_key=X_IDX,
    )
    return wpm


def test_wpm(wpm: DataFrame[WdwPeakMapWide]) -> None:
    pass


@pytest.fixture
def p0(
    wpm: DataFrame[WdwPeakMapWide],
    timestep: float,
    p0_param_cats_fix: pd.CategoricalDtype,
) -> DataFrame[P0]:
    p0 = p0_factory(
        windowed_peak_params=wpm,
        maxima_key=MAXIMA_KEY,
        X_idx_key=X_IDX,
        p0_key=P0_KEY,
        p_idx_key=P_IDX_KEY,
        param_key=PARAM_KEY,
        timestep=timestep,
        w_idx_key=W_IDX_KEY,
        whh_width_key=WHH_WIDTH_KEY,
        skew_key=SKEW_KEY,
        whh_width_half_key=WHH_WIDTH_HALF_KEY,
        p0_param_cat_dtype=p0_param_cats,
        param_val_loc=PARAM_VAL_LOC,
        param_val_maxima=PARAM_VAL_MAX,
        param_val_skew=PARAM_VAL_SKEW,
        param_val_width=PARAM_VAL_WIDTH,
    )

    return p0


def test_p0_factory(p0: DataFrame[P0]) -> None:
    pass


@pytest.fixture
def bounds(
    p0: DataFrame[P0],
    X_windowed,
    timestep: float,
) -> DataFrame[Bounds]:

    default_bounds: DataFrame[Bounds] = bounds_factory(
        p0=p0,
        X_w=X_windowed,
        timestep=timestep,
        whh_width_half_key=WHH_WIDTH_HALF_KEY,
        skew_key=SKEW_KEY,
        X_key=X,
        X_idx_key=X_IDX,
        w_idx_key=W_IDX_KEY,
        w_type_key=W_TYPE_KEY,
        p_idx_key=P_IDX_KEY,
        param_key=PARAM_KEY,
        p0_key=P0_KEY,
        lb_key=LB_KEY,
        ub_key=UB_KEY,
        amp_ub_mult=AMP_UB_MULT,
        amp_lb_mult=AMP_LB_MULT,
        maxima_key=MAXIMA_KEY,
        skew_lb_scalar=SKEW_LB_SCALAR,
        skew_ub_scalar=SKEW_UB_SCALAR,
        param_val_maxima=PARAM_VAL_MAX,
        param_val_loc=PARAM_VAL_LOC,
        param_val_width=PARAM_VAL_WIDTH,
        param_val_skew=PARAM_VAL_SKEW,
        param_cats=p0_param_cats,
    )
    return default_bounds


def test_bounds_factory(bounds: DataFrame[Bounds]) -> None:
    pass


def test_DataPrepper_pipeline(
    peak_map: DataFrame[PeakMap],
    X_windowed,
    timestep: float,
    w_idx_key: str,
    w_type_key: str,
    p_idx_key: str,
    X_idx_key: str,
    X_key: str,
    time_key: str,
    whh_ha: str,
) -> None:

    dp = DataPrepper()

    params = (
        dp.fit(
            pm=peak_map,
            X_w=X_windowed,
            X_key=X_key,
            p_idx_key=p_idx_key,
            time_key=time_key,
            timestep=timestep,
            w_idx_key=w_idx_key,
            w_type_key=w_type_key,
            whh_width_key=WHH_WIDTH_KEY,
            X_idx_key=X_idx_key,
        )
        .transform()
        .params
    )

    Params.validate(params)
    breakpoint()
