"""
TODO:
- [ ] rewrite this testing suite to match the window module rewrite. This is not going to happen unless something goes drastically wrong, so in lieu of that..
    - [ ] write test for MapWindows class transform
"""

from numpy import int64
import polars as pl
import numpy as np
import pandas as pd
import pytest
from pandera.typing.pandas import DataFrame, Series

from hplc_py.map_peaks.map_peaks import MapPeaks, PeakMapWide


from hplc_py.map_windows.map_windows import (
    MapWindows,
    p_wdw_intvl_factory,
    peak_base_intvl_factory,
    map_wdws_to_peaks,
    set_wdw_intvls_from_peak_intvls,
    label_interpeaks,
    set_peak_wndwd_X_idx,
    peak_intvls_as_frame,
    window_X,
)

from hplc_py.hplc_py_typing.hplc_py_typing import (
    PeakWindows,
    X_PeakWindowed,
    X_Windowed,
    X_Schema,
)


@pytest.fixture
def mw() -> MapWindows:
    mw = MapWindows()
    return mw


def test_mw(
    mw: MapWindows,
    X: DataFrame[X_Schema],
) -> None:
    X_w = mw.fit(X=X).transform().X_w
    X_Windowed.validate(X_w, lazy=True)
    breakpoint()
