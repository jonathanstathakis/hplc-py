import pandas as pd
from pandera.typing import DataFrame, Series
import pytest

from hplc_py.deconvolve_peaks.deconvolution import (
    PeakDeconvolver,
    build_peak_report,
    construct_peak_signals,
    reconstruct_signal,
)
from hplc_py.deconvolve_peaks.definitions import (
    RECON_KEY,
    X_KEY,
    WHH_WIDTH_HALF_KEY,
    SKEW_KEY,
    P_IDX_KEY,
    X_IDX_KEY,
    UNMIXED_KEY,
    AREA_UNMIXED_KEY,
    MAXIMA_UNMIXED_KEY,
    RETENTION_TIME_KEY,
    W_IDX_KEY,
)
from hplc_py.deconvolve_peaks.schemas import PReport, PSignals, Popt, RSignal
from hplc_py.map_windows.schemas import X_Windowed


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
def peak_report(
    stored_popt: DataFrame[Popt],
    psignals: DataFrame[PSignals],
) -> DataFrame[PReport]:
    peak_report = build_peak_report(
        popt=stored_popt,
        unmixed_df=psignals,
        amp_key=X_KEY,
        area_unmixed_key=AREA_UNMIXED_KEY,
        maxima_unmixed_key=MAXIMA_UNMIXED_KEY,
        p_idx_key=P_IDX_KEY,
        loc_key=RETENTION_TIME_KEY,
        skew_key=SKEW_KEY,
        unmixed_key=UNMIXED_KEY,
        w_idx_key=W_IDX_KEY,
        whh_half_key=WHH_WIDTH_HALF_KEY,
    )

    return peak_report


TIME_KEY = "time"


@pytest.fixture
def psignals(
    time: Series[float],
    stored_popt: DataFrame[Popt],
    X_w: DataFrame[X_Windowed],
):
    peak_signals = construct_peak_signals(
        X_w=X_w,
        popt_df=stored_popt,
        maxima_key=X_KEY,
        loc_key=TIME_KEY,
        width_key=WHH_WIDTH_HALF_KEY,
        skew_key=SKEW_KEY,
        p_idx_key=P_IDX_KEY,
        X_idx_key=X_IDX_KEY,
        unmixed_key=UNMIXED_KEY,
    )

    return peak_signals


@pytest.fixture
def peak_deconvolver() -> PeakDeconvolver:

    peak_deconvolver = PeakDeconvolver()

    return peak_deconvolver


@pytest.fixture
def r_signal(
    peak_deconvolver: PeakDeconvolver,
    psignals: DataFrame[PSignals],
) -> DataFrame[RSignal]:

    r_signal = reconstruct_signal(
        peak_signals=psignals,
        p_idx_key=P_IDX_KEY,
        X_idx_key=X_IDX_KEY,
        unmixed_key=UNMIXED_KEY,
        recon_key=RECON_KEY,
    )

    return r_signal