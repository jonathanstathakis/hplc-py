"""
TODO:
- [ ] rewrite this testing suite to match the window module rewrite. This is not going to happen unless something goes drastically wrong, so in lieu of that..
    - [ ] write test for MapWindows class transform
"""

import pytest
from pandera.typing.pandas import DataFrame
from hplc_py.common.common_schemas import X_Schema


from hplc_py.map_windows.map_windows import (
    MapWindows,
)

from hplc_py.map_windows.schemas import (
    X_Windowed,
)


@pytest.fixture
def mw() -> MapWindows:
    mw = MapWindows()
    return mw


def test_mw(
    mw: MapWindows,
    X_data,
) -> None:
    X_w = mw.fit(X=X_data).map_windows().X_windowed
    X_Windowed.validate(X_w, lazy=True)
    breakpoint()
