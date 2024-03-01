import pandera as pa
import pytest
from pandera.typing import DataFrame, Series

from hplc_py.common.common_schemas import X_Schema
from hplc_py.map_peaks.schemas import PeakMap
from hplc_py.map_windows.map_windows import MapWindows
from hplc_py.map_windows.schemas import X_Windowed


@pytest.fixture
@pa.check_types
def X_windowed(
    mw: MapWindows,
    X_data,
    timestep: float,
) -> DataFrame[X_Windowed]:

    X_w = mw.fit(X_data, timestep).map_windows().X_windowed

    return X_w


@pytest.fixture
def mw() -> MapWindows:
    mw = MapWindows()
    return mw


@pytest.fixture
def right_bases(
    peak_map: DataFrame[PeakMap],
    pb_right_key: str,
) -> Series[int]:
    right_bases: Series[int] = Series[int](peak_map[pb_right_key], dtype=int)
    return right_bases


@pytest.fixture
def left_bases(
    peak_map: DataFrame[PeakMap],
    pb_left_key: str,
) -> Series[int]:

    left_bases: Series[int] = Series[int](peak_map[pb_left_key])
    return left_bases
