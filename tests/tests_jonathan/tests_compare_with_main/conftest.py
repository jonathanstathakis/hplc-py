import os
from typing import Any

import hplc
import pandas as pd
import polars as pl
import pytest
from numpy import float64, int64
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series

from hplc_py.baseline_correction import CorrectBaseline
from hplc_py.chromatogram import Chromatogram
from hplc_py.deconvolve_peaks.deconvolution import PeakDeconvolver
from hplc_py.fit_assessment import FitAssessment
from hplc_py.hplc_py_typing.hplc_py_typing import Popt, WindowedSignal
from hplc_py.map_peaks.map_peaks import MapPeaks, PeakMapWide
from hplc_py.map_windows.map_windows import MapWindows

from ..tests.conftest import prepare_dataset_for_input


@pytest.fixture
def asschrom_dset():
    path = "tests/test_data/test_assessment_chrom.csv"

    dset = pd.read_csv(path)

    cleaned_dset = prepare_dataset_for_input(dset, "x", "y")

    return cleaned_dset


@pytest.fixture
def asschrom_chm(asschrom_dset: pd.DataFrame):
    asschrom_chm = Chromatogram(
        asschrom_dset.time.to_numpy(float64), asschrom_dset.amp.to_numpy(float64)
    )
    return asschrom_chm


@pytest.fixture
def asschrom_timestep(asschrom_chm):
    return asschrom_chm.timestep


@pytest.fixture
def asschrom_amp_raw(asschrom_chm):
    return asschrom_chm.df_pd.amp


@pytest.fixture
def asschrom_time(asschrom_chm):
    return asschrom_chm._data.time


@pytest.fixture
def asschrom_amp_bcorr(
    cb: CorrectBaseline,
    asschrom_amp_raw: NDArray[float64],
    asschrom_timestep: float64,
) -> Series[float64]:

    bcorr: Series[float64] = (
        cb.fit(asschrom_amp_raw, asschrom_timestep).transform().corrected
    )

    return bcorr


@pytest.fixture
def asschrom_peak_map(
    mp: MapPeaks,
    asschrom_amp_bcorr: Series[float64],
    prom: float,
    asschrom_timestep: float,
    asschrom_time: Series[float64],
) -> DataFrame[PeakMapWide]:
    pm = mw.transform(
        X=asschrom_amp_bcorr,
        X=asschrom_time,
        timestep=asschrom_timestep,
        prominence=prom,
        wlen=None,
        find_peaks_kwargs={},
    )
    return pm


@pytest.fixture
def asschrom_left_bases(
    asschrom_peak_map: DataFrame[PeakMapWide],
) -> Series[int64]:

    left_bases: Series[int64] = Series[int64](
        asschrom_peak_map[PeakMapWide.pb_left_idx], dtype=int64
    )
    return left_bases


@pytest.fixture
def asschrom_right_bases(
    asschrom_peak_map: DataFrame[PeakMapWide],
) -> Series[int64]:
    right_bases: Series[int64] = Series[int64](
        asschrom_peak_map[PeakMapWide.pb_right_idx], dtype=int64
    )
    return right_bases


@pytest.fixture
def asschrom_ws(
    mw: MapWindows,
    asschrom_time: Series[float64],
    asschrom_amp_bcorr: Series[float64],
    asschrom_left_bases: Series[float64],
    asschrom_right_bases: Series[float64],
) -> DataFrame[WindowedSignal]:
    ws = mw.transform(
        asschrom_left_bases,
        asschrom_right_bases,
        asschrom_time,
        asschrom_amp_bcorr,
    )
    breakpoint()

    return ws


@pytest.fixture
def psignals(
    dc: PeakDeconvolver,
    asschrom_time: Series[float64],
    stored_popt: DataFrame[Popt],
):
    psignals = dc._construct_peak_signals(asschrom_time, stored_popt)

    return psignals


@pytest.fixture
def main_chm_asschrom_loaded(
    raw_signal_df: DataFrame,
    amp_col: str,
):
    main_chm = hplc.quant.Chromatogram(
        pd.DataFrame(
            raw_signal_df.rename(
                columns={
                    amp_col: "signal",
                }
            )
        )
    )

    return main_chm


@pytest.fixture
def main_chm_asschrom_fitted(
    main_chm_asschrom_loaded: hplc.quant.Chromatogram,
):
    verbose = False
    main_chm_asschrom_loaded.fit_peaks(verbose=verbose)
    main_chm_asschrom_loaded.assess_fit(verbose=verbose)

    return main_chm_asschrom_loaded


@pytest.fixture
def main_chm_asschrom_score(
    main_chm_asschrom_fitted: hplc.quant.Chromatogram,
):
    main_chm_asschrom_fitted.assess_fit()

    return main_chm_asschrom_fitted.scores


@pytest.fixture
def main_chm_asschrom_fitted_pkpth():
    pkpth = "/Users/jonathan/hplc-py/tests/tests_jonathan/main_chm_asschrom_fitted.pk"

    return pkpth


@pytest.fixture
def main_chm_asschrom_fitted_pk(
    main_chm_asschrom_fitted_pkpth: str,
) -> hplc.quant.Chromatogram:
    if not os.path.isfile(
        "/Users/jonathan/hplc-py/tests/tests_jonathan/main_chm_asschrom_fitted.pk"
    ):
        raise RuntimeError(
            "No main pickle file found. ./tests/tests_jonathan/test_main_asschrom.py::test_pk_main_chm_asschrom_fitted must be run first."
        )

    import pickle

    return pickle.load(open(main_chm_asschrom_fitted_pkpth, "rb"))


@pytest.fixture
def main_params(main_chm_asschrom_fitted_pk):
    """
    Returns the main asschrom curve fit parameters in long form.

    columns: ['w_idx','p_idx','param','bng','val']
    """
    params_: pd.DataFrame = main_chm_asschrom_fitted_pk.params_jono

    params: pl.DataFrame = pl.from_pandas(params_).melt(
        id_vars=["w_idx", "p_idx", "param"],
        value_vars=["lb", "p0", "ub"],
        variable_name="bng",
        value_name="val",
    )

    return params


@pytest.fixture
def main_peak_report(main_chm_asschrom_fitted_pk):
    peaks = (
        pl.DataFrame(main_chm_asschrom_fitted_pk.peaks)
        .drop("peak_id")
        .with_row_index("p_idx")
    )

    return peaks


@pytest.fixture
def main_scores(main_chm_asschrom_fitted_pk):
    scores = pl.LazyFrame(
        main_chm_asschrom_fitted_pk.scores.reset_index(drop=True)
    ).with_columns(window_id=pl.col("window_id") - 1)

    scores = scores.select(
        [
            "window_type",
            "window_id",
            "time_start",
            "time_end",
            "signal_area",
            "inferred_area",
            "signal_variance",
            "signal_mean",
            "signal_fano_factor",
            "reconstruction_score",
        ]
    )

    return scores


@pytest.fixture
def main_window_props_windowed_signal(main_chm_asschrom_fitted_pk: Any):
    """
    Extracts the main asschrom window properties and windowed input signals.
    """
    w_props_ = []
    w_sigs_ = []
    for w_idx, props in main_chm_asschrom_fitted_pk.window_props.items():
        w_idx = int(w_idx) - 1

        long_vars = ["time_range", "signal"]

        w_prop = (
            pl.DataFrame(
                {
                    "w_idx": w_idx,
                    **{k: v for k, v in props.items() if k not in long_vars},
                }
            )
            .with_row_index("p_idx")
            .select(["w_idx", "p_idx", "amplitude", "location", "width"])
        )

        w_props_.append(w_prop)

        w_sig = pl.DataFrame(
            {"w_idx": w_idx, **{k: v for k, v in props.items() if k in long_vars}}
        ).with_row_index("t_idx")

        w_sigs_.append(w_sig)

    w_props = pl.concat(w_props_)
    w_sigs = pl.concat(w_sigs_)

    return w_props, w_sigs


@pytest.fixture
def main_window_props(main_window_props_windowed_signal: tuple[Any, Any]):
    """
    The window properties of the main package asschrom signal as measured by `_assign_windows`
    """
    return main_window_props_windowed_signal[0]


@pytest.fixture
def main_windowed_peak_signals(main_window_props_windowed_signal: tuple[Any, Any]):
    """
    The main package asschrom windowed input signal, not the reconstruction.

    columns: 't_idx','w_idx','time_range','signal'
    """
    return main_window_props_windowed_signal[1]


@pytest.fixture
def main_window_df(main_chm_asschrom_fitted_pk: Any):
    """
    Main asschrom windowed signal - input and corrected, with background.

    Columns: ['window_type', 'window_id', 't_idx', 'time', 'signal', 'signal_corrected', 'estimated_background']
    """
    window_df = (
        pl.DataFrame(main_chm_asschrom_fitted_pk.window_df)
        .select(
            [
                "window_type",
                "window_id",
                "time_idx",
                "time",
                "signal",
                "signal_corrected",
                "estimated_background",
            ]
        )
        .with_columns(window_id=pl.col("window_id") - 1)
        .rename({"time_idx": "t_idx"})
    )
    return window_df


@pytest.fixture
def main_unmixed_signals(main_chm_asschrom_fitted_pk: Any):
    unmixed_signals = (
        pl.DataFrame(main_chm_asschrom_fitted_pk.unmixed_chromatograms)
        .with_row_index("t_idx")
        .melt(id_vars="t_idx", value_name="amp_unmixed", variable_name="p_idx")
        .with_columns(
            p_idx=pl.col("p_idx").str.replace("column_", "").cast(pl.UInt32),
        )
    )
    return unmixed_signals


@pytest.fixture
def main_extract_popt_and_peak_window_recon_signal(main_chm_asschrom_fitted_pk: Any):
    """
    Extracts the peak properties and window reconstructed signals.

    Args:
        peak_props (dict): A dictionary containing the window peak properties.

    Returns:
        tuple: A tuple containing two pandas DataFrames - `popt` and `w_recon`.
            - `popt`: DataFrame containing the extracted peak properties.
            - `w_recon`: DataFrame containing the reconstructed signals of each peak in each window. these signals are reconstructed from the time interval between window [0] and window [n], rather than the whole time series.
    """

    popts = []
    w_recons = []

    long_cols = ["t_range", "reconstructed_signal"]
    for w_idx, peaks in main_chm_asschrom_fitted_pk._peak_props.items():
        w_idx = int(w_idx) - 1
        for p_idx, props in peaks.items():
            p_idx = int(p_idx.split("_")[1]) - 1

            popts.append(
                pl.DataFrame(
                    {
                        "w_idx": w_idx,
                        "p_idx": p_idx,
                        **{k: v for k, v in props.items() if k not in long_cols},
                    }
                )
            )

            w_recons.append(
                pl.DataFrame(
                    {
                        "w_idx": w_idx,
                        "p_idx": p_idx,
                        **{k: v for k, v in props.items() if k in long_cols},
                    }
                )
                .with_columns(
                    t_idx=(pl.col("t_range") * 100).cast(pl.UInt32),
                )
                .rename({"t_range": "time"})
            )

    popt = (
        pl.concat(popts)
        .drop("p_idx")
        .with_row_index("p_idx")
        .select(
            [
                "w_idx",
                "p_idx",
                "amplitude",
                "retention_time",
                "scale",
                "alpha",
                "area",
            ]
        )
    )
    w_recon = pl.concat(w_recons)

    return popt, w_recon


@pytest.fixture
def main_popt(main_extract_popt_and_peak_window_recon_signal: tuple[Any, Any]):
    """
    Main package asschrom popt table.

    present in wide form with columns: ['w_idx','p_idx','amplitude','retention_time', 'scale','alpha','area']
    """
    return main_extract_popt_and_peak_window_recon_signal[0]


@pytest.fixture
def main_peak_window_recon_signal(
    main_extract_popt_and_peak_window_recon_signal: tuple[Any, Any],
):
    """
    Main package asschrom windowed signal reconstructions.

    These are the signals used to calculate the window area in the reconstruction scoring.

    Present with columns: ['t_idx','w_idx','p_idx','w_recon_sig']
    """

    return main_extract_popt_and_peak_window_recon_signal[1]


@pytest.fixture
def main_pm_(main_chm_asschrom_fitted_pk):
    """
    Main package asschrom peak map

    The data gathered from scipy peak widths
    """
    peak_map = main_chm_asschrom_fitted_pk._peak_map_jono

    return peak_map


@pytest.fixture
def main_peak_widths_amp_input(main_chm_asschrom_fitted_pk):
    """
    The amplitude series inputted into scipy peak widths by main. May or may not be the
    same as the output of `correct_baseline`, I have not confirmed that.
    """
    amp_input = main_chm_asschrom_fitted_pk._jono_assign_window_intensity
    return amp_input


@pytest.fixture
def rtol():
    return 1e-2


@pytest.fixture
def ftol():
    return 1e-2


@pytest.fixture
def asschrom_scores(
    fa: FitAssessment,
    asschrom_ws: DataFrame[WindowedSignal],
    rtol: float,
    ftol: float,
) -> DataFrame:
    scores = fa.calc_wdw_aggs(asschrom_ws, rtol, ftol)

    return scores
