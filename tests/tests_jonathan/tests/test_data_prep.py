import pandera as pa
import pytest
from numpy import float64
from pandera.typing import DataFrame

from hplc_py.deconvolve_peaks.prepare_popt_input import DataPrepper

from hplc_py.hplc_py_typing.hplc_py_typing import (
    P0,
    Bounds,
    Params,
    X_Windowed,
    WdwPeakMapWide,
)
from hplc_py.map_peaks.map_peaks import PeakMapWide


@pytest.fixture
def dp():
    dp = DataPrepper()
    return dp


@pytest.fixture
@pa.check_types
def wpm(
    dp: DataPrepper,
    peak_map: DataFrame[PeakMapWide],
    X_w: DataFrame[X_Windowed],
) -> DataFrame[WdwPeakMapWide]:
    wpm = dp._window_peak_map(peak_map, X_w)

    return wpm


@pytest.fixture
def p0(
    dp: DataPrepper,
    wpm: DataFrame[PeakMapWide],
    timestep: float64,
) -> DataFrame[P0]:
    wpm_ = wpm.loc[:, [InP0.w_idx, InP0.p_idx, InP0.amp, InP0.time, InP0.whh]]

    wpm_ = DataFrame[InP0](wpm_)
    p0 = dp._p0_factory(
        wpm_,
        timestep,
    )

    return p0


@pytest.fixture
def default_bounds(
    dp: DataPrepper,
    p0: DataFrame[P0],
    X_w: DataFrame[X_Windowed],
    timestep: float64,
) -> DataFrame[Bounds]:
    default_bounds = dp._bounds_factory(
        p0,
        X_w,
        timestep,
    )
    return default_bounds


@pytest.fixture
def params(
    dp: DataPrepper,
    peak_map: DataFrame[PeakMapWide],
    X_w: DataFrame[X_Windowed],
    timestep: float64,
) -> DataFrame[Params]:
    params = dp.transform(peak_map, X_w, timestep)

    return params


def test_map_peaks_exec(
    peak_map: DataFrame[PeakMapWide],
) -> None:
    PeakMapWide.validate(peak_map, lazy=True)


def test_window_peak_map(
    wpm: DataFrame[WdwPeakMapWide],
) -> None:

    WdwPeakMapWide.validate(wpm, lazy=True)


def test_p0_factory(
    p0: DataFrame[P0],
):
    """
    Test the initial guess factory output against the dataset-specific schema.
    """

    P0.validate(p0, lazy=True)


def test_default_bounds_factory(
    default_bounds: DataFrame[Bounds],
) -> None:
    """
    Define default bounds schemas
    """
    Bounds.validate(default_bounds, lazy=True)


def test_prepare_params(
    params: DataFrame[Params],
) -> None:
    Params.validate(params, lazy=True)



@pytest.fixture
def whh_key()->str:
    return "whh"

def test_DataPrepper_pipeline(
    peak_map: DataFrame[PeakMapWide],
    X_w: DataFrame[X_Windowed],
    timestep: float,
    w_idx_key: str,
    w_type_key: str,
    p_idx_key: str,
    X_idx_key: str,
    X_key: str,
    time_key: str,
    whh_key: str,
) -> None:

    dp = DataPrepper()
    
    params = (dp.fit(
        pm=peak_map,
        X_w=X_w,
        X_key=X_key,
        p_idx_key=p_idx_key,
        time_key=time_key,
        timestep=timestep,
        w_idx_key=w_idx_key,
        w_type_key=w_type_key,
        whh_key=whh_key,
        X_idx_key=X_idx_key,
        )
    .transform()
    .params
    )
    breakpoint()