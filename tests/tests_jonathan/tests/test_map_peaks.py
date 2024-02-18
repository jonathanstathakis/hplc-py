from dataclasses import dataclass, asdict
from numpy import int64
from numpy.typing import NDArray

import polars as pl


import pytest
from pandera.typing import DataFrame
import pandera as pa
from hplc_py.common_schemas import X_Schema

from hplc_py.hplc_py_typing.hplc_py_typing import (
    WHH,
    FindPeaks,
    PeakBases,
    PeakMapWide,
)

from hplc_py.map_peaks.map_peaks import (
    MapPeaks,
    PPD,
    set_findpeaks,
    get_peak_prom_data,
    width_df_factory,
)


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
def fp_cols():
    return {
        "X_key": "X",
        "X_idx_key": "X_idx",
        "p_idx_key": "p_idx",
        "prom_key": "prom",
        "prom_lb_key": "prom_left",
        "prom_rb_key": "prom_right",
        "maxima_key": "maxima",
    }


@pytest.fixture
@pa.check_types
def fp(
    X: DataFrame[X_Schema],
    fp_cols: dict[str, str],
    prom: float,
    wlen: None,
) -> DataFrame[FindPeaks]:
    fp = set_findpeaks(
        X=X,
        prominence=prom,
        wlen=wlen,
        **fp_cols,
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


@dataclass
class FP_Keys:
    prom_key: str = "prom"
    prom_lb_key: str = "prom_left"
    prom_rb_key: str = "prom_right"


@pytest.fixture
def fp_keys():
    fp_keys = FP_Keys()

    return fp_keys


@pytest.fixture
def ppd(
    fp: DataFrame[FindPeaks],
    fp_keys: FP_Keys,
) -> PPD:
    ppd = get_peak_prom_data(fp=fp, **asdict(fp_keys))
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
        width_df_factory(
            X=X,
            peak_t_idx=pt_idx,
            peak_prom_data=ppd,
            rel_height=whh_rel_height,
            prefix="whh",
            p_idx_key="p_idx",
            X_key="X",
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
    pb_ = width_df_factory(
        X=X,
        peak_t_idx=pt_idx,
        peak_prom_data=ppd,
        rel_height=pb_rel_height,
        prefix="pb",
        p_idx_key='p_idx',
        X_key='X'
    )

    pb = DataFrame[PeakBases](pb_)

    return pb
