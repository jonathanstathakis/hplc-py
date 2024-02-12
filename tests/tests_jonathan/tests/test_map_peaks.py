from matplotlib.figure import Figure
from numpy import float64, int64
from numpy.typing import NDArray

import polars as pl
import polars.selectors as ps
import polars.testing as polt

from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandera.typing import Series, DataFrame
import pandera as pa

from hplc_py.baseline_correct.correct_baseline import SignalDFBCorr

from hplc_py.io_validation import IOValid

from hplc_py.hplc_py_typing.hplc_py_typing import (
    WHH,
    FindPeaks,
    PeakBases,
    PeakMapWide,
    X_Schema,
)

from hplc_py.map_signals.map_peaks.map_peaks import MapPeaks, PPD
from hplc_py.map_signals.map_peaks.map_peaks_viz import PeakMapWideViz
from hplc_py.show import SignalPlotter
from matplotlib.axes import Axes

pl.Config(set_tbl_cols=50)


def test_set_fp(
    fp: DataFrame[FindPeaks],
):
    fp_ = fp.reset_index(drop=True).rename_axis(index="idx")

    try:
        FindPeaks.validate(fp_, lazy=True)
    except pa.errors.SchemaError as e:
        e.add_note(f"\n{e.data}")
        e.add_note(f"\n{e.failure_cases}")


def test_set_whh(
    whh: DataFrame[WHH],
) -> None:
    WHH(whh)


def test_set_pb(
    pb: DataFrame[PeakBases],
) -> None:
    PeakBases(pb)


def test_map_peaks(
    peak_map: DataFrame[PeakMapWide],
) -> None:
    PeakMapWide(peak_map, lazy=True)


#######################################


@pytest.fixture
def prom() -> float:
    return 0.01


@pytest.fixture
def wlen() -> None:
    return None


@pytest.fixture
def fp(
    X: DataFrame[X_Schema],
    prom: float,
    mp: MapPeaks,
    wlen: None,
) -> DataFrame[FindPeaks]:
    fp = mp._set_findpeaks(
        X=X,
        prominence=prom,
        wlen=wlen,
    )

    return fp


@pytest.fixture
def whh_rel_height() -> float:
    return 0.5


@pytest.fixture
def pb_rel_height() -> float:
    return 1.0


@pytest.fixture
def pt_idx_col():
    return str(FindPeaks.X_idx)


@pytest.fixture
def pt_idx(
    fp: DataFrame[FindPeaks],
    pt_idx_col: str,
) -> NDArray[int64]:
    return fp[pt_idx_col].to_numpy(int64)


@pytest.fixture
def ppd(mp: MapPeaks, fp: DataFrame[FindPeaks]) -> PPD:
    ppd = mp.get_peak_prom_data(fp)
    return ppd


@pytest.fixture
def whh(
    mp: MapPeaks,
    X: DataFrame[X_Schema],
    pt_idx: NDArray[int64],
    ppd: PPD,
    whh_rel_height: float,
) -> DataFrame[WHH]:
    whh = DataFrame[WHH](
        mp.width_df_factory(
            X=X,
            peak_t_idx=pt_idx,
            peak_prom_data=ppd,
            rel_height=whh_rel_height,
            prefix="whh",
        )
    )

    return whh


@pytest.fixture
def pb(
    mp: MapPeaks,
    X: DataFrame[X_Schema],
    pt_idx: NDArray[int64],
    pb_rel_height: float,
    ppd: PPD,
) -> DataFrame[PeakBases]:
    """
    The peak bases
    """
    pb_ = mp.width_df_factory(
        X=X,
        peak_t_idx=pt_idx,
        peak_prom_data=ppd,
        rel_height=pb_rel_height,
        prefix="pb",
    )

    pb = DataFrame[PeakBases](pb_)

    return pb
