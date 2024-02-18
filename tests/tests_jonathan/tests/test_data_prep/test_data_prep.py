import pandas as pd
import pytest
from pandera.typing import DataFrame

from hplc_py.deconvolve_peaks.definitions import (
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
    WB_KEY,
    WHH_WIDTH_HALF_KEY,
    WHH_WIDTH_KEY,
    X_IDX_KEY,
    X_KEY,
    SKEW_LB_SCALAR,
    SKEW_UB_SCALAR,
    PARAM_VAL_MAX,
    PARAM_VAL_LOC,
    PARAM_VAL_WIDTH,
    PARAM_VAL_SKEW,
)
from hplc_py.deconvolve_peaks.prepare_popt_input import (
    DataPrepper,
    bounds_factory,
    p0_factory,
    window_peak_map,
)
from hplc_py.deconvolve_peaks.schemas import P0, Bounds, InP0, Params
from hplc_py.deconvolve_peaks.typing import p0_param_cats
from hplc_py.hplc_py_typing.hplc_py_typing import (
    WdwPeakMapWide,
)
from hplc_py.map_peaks.map_peaks import PeakMapWide
from hplc_py.map_windows.schemas import X_Windowed


@pytest.fixture
def p0_param_cats_fix():
    return p0_param_cats


@pytest.fixture
def dp():
    dp = DataPrepper()
    return dp


@pytest.fixture
def wpm(
    peak_map: DataFrame[PeakMapWide],
    X_w: DataFrame[X_Windowed],
) -> DataFrame[WdwPeakMapWide]:
    wpm: DataFrame[WdwPeakMapWide] = window_peak_map(
        peak_map=peak_map,
        X_w=X_w,
        t_idx_key=X_IDX_KEY,
        w_idx_key=W_IDX_KEY,
        w_type_key=W_TYPE_KEY,
        X_idx_key=X_IDX_KEY,
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
        wpm=wpm,
        maxima_key=MAXIMA_KEY,
        X_idx_key=X_IDX_KEY,
        p0_key=P0_KEY,
        p_idx_key=P_IDX_KEY,
        param_key=PARAM_KEY,
        timestep=timestep,
        w_idx_key=W_IDX_KEY,
        whh_width_key=WHH_WIDTH_KEY,
        skew_key=SKEW_KEY,
        whh_width_half_key=WHH_WIDTH_HALF_KEY,
        p0_param_cat_dtype=p0_param_cats,
    )

    return p0


def test_p0_factory(p0: DataFrame[P0]) -> None:
    pass


@pytest.fixture
def bounds(
    p0: DataFrame[P0],
    X_w: DataFrame[X_Windowed],
    timestep: float,
) -> DataFrame[Bounds]:
    default_bounds: DataFrame[Bounds] = bounds_factory(
        p0=p0,
        X_w=X_w,
        timestep=timestep,
        whh_width_half_key=WHH_WIDTH_HALF_KEY,
        skew_key=SKEW_KEY,
        X_key=X_KEY,
        X_idx_key=X_IDX_KEY,
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
    )
    return default_bounds


def test_bounds_factory(bounds: DataFrame[Bounds]) -> None:
    pass


# @pytest.fixture
# def params(
#     dp: DataPrepper,
#     peak_map: DataFrame[PeakMapWide],
#     X_w: DataFrame[X_Windowed],
#     timestep: float64,
# ) -> DataFrame[Params]:
#     params = dp.transform(peak_map, X_w, timestep)

#     return params


# def test_DataPrepper_pipeline(
#     peak_map: DataFrame[PeakMapWide],
#     X_w: DataFrame[X_Windowed],
#     timestep: float,
#     w_idx_key: str,
#     w_type_key: str,
#     p_idx_key: str,
#     X_idx_key: str,
#     X_key: str,
#     time_key: str,
#     whh_ha: str,
# ) -> None:

#     dp = DataPrepper()

#     params = (
#         dp.fit(
#             pm=peak_map,
#             X_w=X_w,
#             X_key=X_key,
#             p_idx_key=p_idx_key,
#             time_key=time_key,
#             timestep=timestep,
#             w_idx_key=w_idx_key,
#             w_type_key=w_type_key,
#             whh_key=whh_key,
#             X_idx_key=X_idx_key,
#         )
#         .transform()
#         .params
#     )
#     breakpoint()
