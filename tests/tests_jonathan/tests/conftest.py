import polars as pl
import numpy as np
import pandera as pa

import pandas as pd
import pytest
from numpy import float64, int64
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series

from hplc_py.baseline_correction import CorrectBaseline
from hplc_py.deconvolve_peaks.deconvolution import PeakDeconvolver
from hplc_py.hplc_py_typing.custom_checks import col_a_less_than_col_b  # noqa: F401
from hplc_py.hplc_py_typing.hplc_py_typing import (
    Popt,
    PReport,
    PSignals,
    RawData,
    RSignal,
    X_Schema,
)

from hplc_py.map_windows.typing import X_Windowed

from hplc_py.map_peaks.map_peaks import MapPeaks, PeakMapWide
from hplc_py.map_windows.map_windows import MapWindows
from hplc_py.fit_assessment import FitAssessment
from hplc_py.misc.misc import compute_timestep


def prepare_dataset_for_input(
    data: pd.DataFrame,
    time_col: str,
    amp_col: str,
) -> DataFrame[RawData]:
    """
    Rename the x and y columns to match SignalDFLoaded schema.

    :param data: the dirty dataset
    :type data: pd.DataFrame
    :param time_col: the x column to be renamed to match the schema
    :type time_col: str
    :param amp_col: the y column to be renamed to match the schema
    :type time_col: str
    """
    data = data.rename(
        {time_col: RawData.time, amp_col: RawData.amp}, axis=1, errors="raise"
    ).reset_index(names="t_idx")

    data = DataFrame[RawData](data)

    return data


@pytest.fixture
def ringland_shz_dset():
    path = "tests/tests_jonathan/test_data/a0301_2021_chris_ringland_shiraz.csv"

    dset = pd.read_csv(path)
    dset = dset[["time", "signal"]]
    cleaned_dset = prepare_dataset_for_input(dset, "time", "signal")
    return cleaned_dset


@pytest.fixture
def asschrom_dset():
    path = "tests/test_data/test_assessment_chrom.csv"

    dset = pd.read_csv(path)

    cleaned_dset = prepare_dataset_for_input(dset, "x", "y")

    return cleaned_dset


# @pytest.fixture
# @pa.check_types
# def in_signal(
#     asschrom_dset,
#     ) -> DataFrame[RawData]:
#     return asschrom_dset
@pytest.fixture
@pa.check_types
def in_signal(
    ringland_shz_dset,
) -> DataFrame[RawData]:
    return ringland_shz_dset


@pytest.fixture
def time_colname():
    return "time"


@pytest.fixture
def amp_col():
    return "amp"


@pytest.fixture
def time(in_signal: DataFrame[RawData]) -> Series[float64]:
    return Series[float64](in_signal["time"])


@pytest.fixture
def timestep(time: NDArray[float64]) -> float64:
    timestep = compute_timestep(time)
    return timestep


@pytest.fixture
def cb():
    cb = CorrectBaseline(window_size=0.65)
    return cb


@pytest.fixture
def windowsize():
    return 5


@pytest.fixture
def amp_colname(amp_col: str) -> str:
    bcorr_col_str: str = amp_col
    return bcorr_col_str


@pytest.fixture
def background_colname():
    return "background"


@pytest.fixture
def background(bcorrected_signal_df, background_colname):
    return bcorrected_signal_df[background_colname]


@pytest.fixture
def peak_deconvolver(
) -> PeakDeconvolver:
    
    peak_deconvolver = PeakDeconvolver()
    
    return peak_deconvolver

@pytest.fixture
def int_col():
    return "amp_corrected"


@pytest.fixture
def psignals(
    dc: PeakDeconvolver,
    time: Series[float64],
    stored_popt: DataFrame[Popt],
):
    psignals = dc._construct_peak_signals(time, stored_popt)

    return psignals


@pytest.fixture
def peak_report(
    dc: PeakDeconvolver,
    stored_popt: DataFrame[Popt],
    psignals: DataFrame[PSignals],
) -> DataFrame[PReport]:
    peak_report = dc._get_peak_report(
        stored_popt,
        psignals,
    )

    return peak_report


@pytest.fixture
def popt_parqpath():
    """
    Intended to be used to store a popt df as it is computationally expensive to deconvolute many-peaked windows
    """
    return "/Users/jonathan/hplc-py/tests/tests_jonathan/asschrompopt.parquet"


@pytest.fixture()
def stored_popt(popt_parqpath):
    """
    Read the stored popt_df, short circuiting the slow optimization process
    """
    return pd.read_parquet(popt_parqpath)

@pytest.fixture
def pb_left_key(
):
    return "pb_left"

@pytest.fixture
def pb_right_key(
):
    return "pb_right"

@pytest.fixture
def left_bases(
    peak_map: DataFrame[PeakMapWide],
    pb_left_key: str,
) -> Series[int64]:

    left_bases: Series[int64] = Series[int64](
        peak_map[pb_left_key], dtype=int64
    )
    return left_bases


@pytest.fixture
def right_bases(
    peak_map: DataFrame[PeakMapWide],
    pb_right_key: str,
) -> Series[int64]:
    right_bases: Series[int64] = Series[int64](
        peak_map[pb_right_key], dtype=int64
    )
    return right_bases


@pytest.fixture
def mw() -> MapWindows:
    mw = MapWindows()
    return mw


@pytest.fixture
def prom() -> float:
    return 0.01


@pytest.fixture
def r_signal(
    dc: PeakDeconvolver,
    psignals: DataFrame[PSignals],
) -> DataFrame[RSignal]:

    r_signal = dc._reconstruct_signal(psignals)

    return r_signal


@pytest.fixture
def fa() -> FitAssessment:
    fa = FitAssessment()
    return fa


@pytest.fixture
def scores(
    fa: FitAssessment,
    asschrom_ws: DataFrame[X_Windowed],
    rtol: float,
    ftol: float,
) -> DataFrame:
    scores = fa.calc_wdw_aggs(asschrom_ws, rtol, ftol)

    return scores


@pytest.fixture
def amp_raw(
    in_signal: DataFrame[RawData],
    in_signal_amp_col: str,
) -> Series[float64]:
    amp = in_signal[in_signal_amp_col]
    return amp


@pytest.fixture
def amp_bcorr(
    cb: CorrectBaseline,
    amp_raw: NDArray[float64],
    timestep: float64,
) -> Series[float64]:

    background = cb.fit(amp_raw, timestep).transform().background

    bcorr = np.subtract(amp_raw, background)
    return bcorr


@pytest.fixture
def in_signal_amp_col() -> str:
    return "amp"


@pytest.fixture
def mp() -> MapPeaks:
    pm = MapPeaks(prominence=0.01)
    return pm


@pytest.fixture
def peak_map(
    mp: MapPeaks,
    X: DataFrame[X_Schema],
) -> DataFrame[PeakMapWide]:

    mp.fit(
        X=X,
    )
    mp.transform()

    pm = mp.peak_map
    return pm


@pytest.fixture
def X(
    amp_bcorr: Series[float64],
) -> DataFrame[X_Schema]:
    X_ = amp_bcorr.to_frame(name="X").reset_index(names="X_idx").astype({"X_idx": int})

    X = DataFrame[X_Schema](X_)
    return X


@pytest.fixture
@pa.check_types
def X_w(
    mw: MapWindows,
    X: DataFrame[X_Schema],
    timestep: float,
) -> DataFrame[X_Windowed]:

    X_w = mw.fit(X, timestep).transform().X_w

    return X_w


@pytest.fixture
def X_idx(X: DataFrame[X_Schema], mw: MapWindows)->pl.DataFrame:
    X_idx = pl.DataFrame({X_Schema.X: np.arange(0, len(X), 1)})

    return X_idx


@pytest.fixture
def p_idx_key() -> str:
    return "p_idx"


@pytest.fixture
def X_key()->str:
    return "X"

@pytest.fixture
def X_idx_key() -> str:
    return "X_idx"


@pytest.fixture
def w_type_key() -> str:
    return "w_type"


@pytest.fixture
def w_idx_key() -> str:
    return "w_idx"


@pytest.fixture
def time_key() -> str:
    return "time"
