from hplc_py.common.common_schemas import X_Schema

import pandas as pd
from pandera.typing import DataFrame, Series
import pytest

from hplc_py.deconvolution.deconvolution import (
    PeakDeconvolver,
    build_peak_report,
    construct_unmixed_signals,
    reconstruct_signal,
)
from hplc_py.common.definitions import (
    RECON_KEY,
    X,
    WHH_WIDTH_HALF_KEY,
    SKEW_KEY,
    P_IDX_KEY,
    X_IDX,
    RECON_KEY,
    AREA_UNMIXED_KEY,
    MAXIMA_UNMIXED_KEY,
    RETENTION_TIME_KEY,
    W_IDX_KEY,
)
from hplc_py.deconvolution.schemas import PReport, PSignals, Popt, RSignal
from hplc_py.map_windows.schemas import X_Windowed


@pytest.fixture
def peak_report(
    popt: DataFrame[Popt],
    psignals: DataFrame[PSignals],
) -> DataFrame[PReport]:
    peak_report = build_peak_report(
        popt=popt,
        unmixed_df=psignals,
        amp_key=X,
        area_unmixed_key=AREA_UNMIXED_KEY,
        maxima_unmixed_key=MAXIMA_UNMIXED_KEY,
        p_idx_key=P_IDX_KEY,
        loc_key=RETENTION_TIME_KEY,
        skew_key=SKEW_KEY,
        unmixed_key=RECON_KEY,
        w_idx_key=W_IDX_KEY,
        whh_half_key=WHH_WIDTH_HALF_KEY,
    )

    return peak_report


TIME_KEY = "time"


@pytest.fixture
def psignals(
    time: Series[float],
    stored_popt: DataFrame[Popt],
    X_windowed,
):
    peak_signals = construct_unmixed_signals(
        X_w=X_windowed,
        popt=stored_popt,
        maxima_key=X,
        loc_key=TIME_KEY,
        width_key=WHH_WIDTH_HALF_KEY,
        skew_key=SKEW_KEY,
        p_idx_key=P_IDX_KEY,
        X_idx_key=X_IDX,
        unmixed_key=RECON_KEY,
    )

    return peak_signals


@pytest.fixture
def peak_deconvolver() -> PeakDeconvolver:

    peak_deconvolver = PeakDeconvolver()

    return peak_deconvolver


@pytest.fixture
def r_signal(
    psignals: DataFrame[PSignals],
) -> DataFrame[RSignal]:

    r_signal = reconstruct_signal(
        peak_signals=psignals,
        p_idx_key=P_IDX_KEY,
        X_idx_key=X_IDX,
        unmixed_key=RECON_KEY,
        recon_key=RECON_KEY,
    )

    return r_signal


@pytest.fixture
def pdc_tform(
    X_data,
    timestep: float,
) -> PeakDeconvolver:
    pdc = PeakDeconvolver(
        which_opt="jax",
        which_fit_func="jax",
    )
    pdc.fit(
        X=X_data,
        timestep=timestep,
    )
    pdc.transform()

    return pdc
