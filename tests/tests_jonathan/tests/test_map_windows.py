from numpy import float64, int64
import polars as pl
import numpy as np
import pandas as pd
import pytest
from pandera.typing.pandas import DataFrame, Series
import pandera as pa
import matplotlib.pyplot as plt

from hplc_py.map_signals.map_peaks.map_peaks import MapPeaks, PeakMapWide

from hplc_py.show import DrawPeakWindows, SignalPlotter

from hplc_py.map_signals.map_windows import (
    MapWindows,
)

from hplc_py.hplc_py_typing.hplc_py_typing import (
    PeakWindows,
    X_PeakWindowed,
    WindowedSignal,
    X_Windowed,
    SignalDFBCorr,
    X_Schema,
)


@pytest.fixture
def mw() -> MapWindows:
    mw = MapWindows()
    return mw


@pytest.fixture
def X_idx(X: DataFrame[X_Schema], mw: MapWindows):
    X_idx = pl.DataFrame({mw._X_idx_colname: np.arange(0, len(X), 1)})

    return X_idx


@pytest.fixture
def left_bases(
    peak_map: DataFrame[PeakMapWide],
    mp: MapPeaks,
) -> Series[int64]:
    left_bases: Series[int64] = Series[int64](peak_map[str(mp._pb_l_col)], dtype=int64)
    return left_bases


@pytest.fixture
def right_bases(
    peak_map: DataFrame[PeakMapWide],
    mp: MapPeaks,
) -> Series[int64]:
    right_bases: Series[int64] = Series[int64](peak_map[mp._pb_r_col], dtype=int64)
    return right_bases


class Test_Peak_W_Intvl_Factory:
    """
    Test the inner workings of `_peak_w_intvl_factory`
    """

    def test_peak_base_intvl_factory(
        self,
        pb_intvls: Series[pd.Interval],
    ) -> None:
        if not isinstance(pb_intvls, pd.Series):
            raise TypeError("expected pd.Series")
        elif pb_intvls.empty:
            raise ValueError("intvls is empty")
        elif not pb_intvls.index.dtype == np.int64:
            raise TypeError(f"Expected np.int64 index, got {pb_intvls.index.dtype}")
        elif not pb_intvls.index[0] == 0:
            raise ValueError("Expected interval to start at 0")

    def test_map_wdws_to_peaks(
        self,
        wdw_peak_mapping: dict[int, list[int]],
    ) -> None:
        if not wdw_peak_mapping:
            raise ValueError("w_idxs is empty")
        if not any(v for v in wdw_peak_mapping.values()):
            raise ValueError("a w_idx is empty")
        if not isinstance(wdw_peak_mapping, dict):
            raise TypeError("expected dict")
        elif not all(isinstance(l, list) for l in wdw_peak_mapping.values()):
            raise TypeError("expected list")
        elif not all(isinstance(x, int) for v in wdw_peak_mapping.values() for x in v):
            raise TypeError("expected values of window lists to be int")

    def test_set_wdw_intvls_from_peak_intvls(
        self, peak_wdw_intvls: dict[str, pd.Interval]
    ) -> None:
        assert isinstance(peak_wdw_intvls, dict)
        for k, v in peak_wdw_intvls.items():
            assert isinstance(k, int)
            assert isinstance(v, pd.Interval)

    def test_p_wdw_intvl_factory(
        self,
        peak_wdw_intvls: dict[str, pd.Interval],
        mw: MapWindows,
        left_bases: Series[int64],
        right_bases: Series[int64],
    ) -> None:
        """
        Compare the output of the top level function with the output of the  unit tested
        inner function calls.
        """
        # the dict generated from the combination of inner functions
        peak_wdw_intvls_1 = mw._p_wdw_intvl_factory(
            left_bases=left_bases, right_bases=right_bases
        )

        # the RHS is the fixture generated from running each function in the test
        # environment.
        assert peak_wdw_intvls_1 == peak_wdw_intvls


@pytest.fixture
def pb_intvls(
    mw: MapWindows,
    left_bases: Series[int64],
    right_bases: Series[int64],
) -> Series[pd.Interval]:
    """
    A pd Series of pd.Interval objects representing the calculated peak windows
    """
    pb_intvls: Series[pd.Interval] = mw._peak_base_intvl_factory(
        left_bases, right_bases
    )
    return pb_intvls


@pytest.fixture
def wdw_peak_mapping(
    mw: MapWindows,
    pb_intvls: Series[pd.Interval],
) -> dict[int, list[int]]:
    w_idxs: dict[int, list[int]] = mw._map_wdws_to_peaks(pb_intvls)
    return w_idxs


@pytest.fixture
def peak_wdw_intvls(
    mw: MapWindows,
    pb_intvls: Series[pd.Interval],
    wdw_peak_mapping: dict[int, list[int]],
) -> dict[int, pd.Interval]:
    """
    W_intvls is a dict of w_idx keys and pd.Interval objects representing peak windows
    """
    w_intvls: dict[int, pd.Interval] = mw._set_wdw_intvls_from_peak_intvls(
        pb_intvls, wdw_peak_mapping
    )
    return w_intvls


@pytest.fixture
def X_idx_pw(
    mw: MapWindows,
    peak_wdw_intvls: dict[int, pd.Interval],
) -> DataFrame[PeakWindows]:

    X_idx_w: DataFrame[PeakWindows] = mw._peak_intvls_as_frame(
        peak_wdw_intvls=peak_wdw_intvls
    )

    return X_idx_w


@pytest.fixture
def X_pw(
    mw: MapWindows, X: DataFrame[X_Schema], X_idx_pw: DataFrame[PeakWindows]
) -> DataFrame[X_PeakWindowed]:
    X_pw = mw._set_peak_wndwd_X_idx(X=X, X_idx_pw=X_idx_pw)
    return X_pw


@pytest.fixture
def X_w(mw: MapWindows, X_pw: DataFrame[X_PeakWindowed]) -> DataFrame[X_Windowed]:
    X_w: DataFrame[X_Windowed] = mw._label_interpeaks(X_pw)

    return X_w


class Test_Window_X:
    """
    test the inner workings of `window_X`
    """

    def test_peak_intvls_as_frame(
        self,
        X_idx_pw,
    ) -> None:
        PeakWindows.validate(X_idx_pw, lazy=True)

    def test_set_peak_wndwd_X_idx(self, X_pw: DataFrame[X_PeakWindowed]) -> None:
        X_PeakWindowed.validate(X_pw, lazy=True)

    def test_label_interpeaks(self, X_w: DataFrame[X_Windowed]) -> None:

        X_Windowed.validate(X_w, lazy=True)

    def test_window_X(
        self,
        mw: MapWindows,
        X: DataFrame[X_Schema],
        peak_wdw_intvls: dict[int, pd.Interval],
        X_w: DataFrame[X_Windowed],
    ) -> None:
        X_w_1 = mw._window_X(X=X, peak_wdw_intvls=peak_wdw_intvls)

        assert X_w_1.equals(X_w)
