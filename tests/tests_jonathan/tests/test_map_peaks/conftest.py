import polars as pl
import pytest
from pandera.typing import DataFrame

from hplc_py.common.common_schemas import X_Schema
from hplc_py.map_peaks.schemas import PeakMapOutput
from hplc_py.map_peaks.map_peaks import MapPeaks
from hplc_py.baseline_correction.baseline_correction import BaselineCorrection
from hplc_py.common import definitions as com_defs
from hplc_py.baseline_correction import definitions as bc_defs


@pytest.fixture
def n_iter():
    return 250


@pytest.fixture
def window_size():
    return 5


@pytest.fixture
def X_bcorr(
    X_data_raw: DataFrame[X_Schema],
    n_iter: int,
    window_size: int,
) -> DataFrame[X_Schema]:
    """
    Return the corrected signal as per the `X_Schema` to allow older tests to function. 2024-02-23 15:41:22
    """

    bc = BaselineCorrection(n_iter=n_iter, window_size=window_size, verbose=True)
    bc.fit(X_data_raw.pipe(pl.from_pandas).select(com_defs.X).to_series().to_numpy())

    X_bcorr = (
        bc.correct_baseline()
        .pipe(pl.from_pandas)
        .pivot(index=["X_idx"], values=["amp"], columns=["signal"])
        .select(pl.col([com_defs.IDX, bc_defs.KEY_CORRECTED]))
        .rename({bc_defs.KEY_CORRECTED: com_defs.X})
        .to_pandas()
        .pipe(X_Schema.validate, lazy=True)
        .pipe(DataFrame[X_Schema])
    )
    return X_bcorr


@pytest.fixture
def map_peaks_mapped(
    X_bcorr: DataFrame[X_Schema],
) -> MapPeaks:
    map_peaks = MapPeaks(
        X=X_bcorr,
    )
    return map_peaks
