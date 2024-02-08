import pandera as pa
import pytest
from numpy import float64
from pandera.typing import DataFrame

from hplc_py.deconvolve_peaks.deconvolution import (
    DataPrepper,
    InP0,
    WdwPeakMap,
)
from hplc_py.hplc_py_typing.hplc_py_typing import (
    P0,
    Bounds,
    Params,
    X_Windowed,
)
from hplc_py.map_signals.map_peaks.map_peaks import PeakMap

def test_map_peaks_exec(
    peak_map: DataFrame[PeakMap],
) -> None:
    PeakMap.validate(peak_map, lazy=True)


def test_window_peak_map(
    wpm: DataFrame[WdwPeakMap],
) -> None:
    
    WdwPeakMap.validate(wpm, lazy=True)


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
def dp():
    dp = DataPrepper()
    return dp

@pytest.fixture
@pa.check_types
def wpm(
    dp: DataPrepper,
    peak_map: DataFrame[PeakMap],
    X_w: DataFrame[X_Windowed],
) -> DataFrame[WdwPeakMap]:
    wpm = dp._window_peak_map(peak_map, X_w)

    return wpm


@pytest.fixture
def p0(
    dp: DataPrepper,
    wpm: DataFrame[PeakMap],
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
    peak_map: DataFrame[PeakMap],
    X_w: DataFrame[X_Windowed],
    timestep: float64,
) -> DataFrame[Params]:
    params = dp.transform(peak_map, X_w, timestep)

    return params
