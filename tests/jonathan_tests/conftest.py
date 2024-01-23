from typing import Any
import polars as pl

import pytest
import hplc
from pandera.typing import DataFrame
import typing
import pandas as pd
from scipy.optimize._lsq.common import in_bounds  # type: ignore
import os

import pytest
import pandera as pa
from pandera.typing import Series, DataFrame
import pandas as pd

import numpy as np

from hplc_py.hplc_py_typing.hplc_py_typing import (
    SignalDFInBase,
    SignalDFInAssChrom,
    PReport,
    P0,
    Bounds,
    PSignals,
    Popt,
    Params,
    isArrayLike,
    FloatArray,
)

from hplc_py.quant import Chromatogram
from hplc_py.baseline_correct.correct_baseline import CorrectBaseline, SignalDFBCorr
from hplc_py.map_signals.map_peaks import MapPeaks
from hplc_py.deconvolve_peaks.mydeconvolution import DataPrepper, PeakDeconvolver
from hplc_py.misc.misc import TimeStep, LoadData
import json


@pytest.fixture
def datapath():
    return "tests/test_data/test_assessment_chrom.csv"


@pytest.fixture
def in_signal(datapath: str) -> DataFrame[SignalDFInBase]:
    data = pd.read_csv(datapath)

    data = data.rename({"x": "time", "y": "amp_raw"}, axis=1, errors="raise")
    data = DataFrame[SignalDFInBase](data)
    return data


def test_in_signal_matches_schema(in_signal: DataFrame[SignalDFInBase]) -> None:
    "currently in signal is asschrom"
    SignalDFInAssChrom(in_signal)
    return None


@pytest.fixture
def chm():
    return Chromatogram()


@pytest.fixture
def time(in_signal: DataFrame[SignalDFInBase]):
    return Series(in_signal["time"], dtype=pd.Float64Dtype())


@pytest.fixture
def ts():
    ts = TimeStep()
    return ts


@pytest.fixture
def timestep(ts: TimeStep, time: FloatArray) -> np.float64:
    timestep: np.float64 = ts.compute_timestep(Series(time, dtype=pd.Float64Dtype()))

    return timestep


@pytest.fixture
def amp_raw(in_signal: DataFrame[SignalDFInBase]):
    amp = in_signal["amp_raw"].values

    return amp


@pytest.fixture
def windowsize():
    return 5


@pytest.fixture
def bcorr_colname(amp_col: str) -> str:
    bcorr_col_str: str = amp_col.replace("raw", "corrected")
    return bcorr_col_str


@pytest.fixture
def timecol():
    return "time"


@pytest.fixture
def amp_col():
    return "amp_raw"


@pytest.fixture
def ld():
    ld = LoadData()
    return ld


@pytest.fixture
def loaded_signal_df(
    ld: LoadData,
    in_signal_df: DataFrame[SignalDFInBase],
):
    lsd = ld.set_signal_df(in_signal_df)

    return lsd


@pytest.fixture
def cb():
    cb = CorrectBaseline()
    return cb


@pytest.fixture
def loaded_cb(
    cb: CorrectBaseline,
    in_signal: DataFrame[SignalDFInBase],
):
    cb.set_signal_df(in_signal)

    return cb


@pytest.fixture
def bcorred_cb(
    loaded_cb: CorrectBaseline,
):
    loaded_cb.correct_baseline()

    return loaded_cb


@pytest.fixture
def bcorred_signal_df(bcorred_cb: CorrectBaseline) -> DataFrame[SignalDFBCorr]:
    bcorred_signal_df = DataFrame[SignalDFBCorr](bcorred_cb._signal_df)

    return bcorred_signal_df


@pytest.fixture
def amp_bcorr(bcorred_signal_df: DataFrame, bcorr_colname: str):
    return bcorred_signal_df[bcorr_colname].to_numpy(np.float64)


@pytest.fixture
def background_colname():
    return "background"


@pytest.fixture
def background(bcorrected_signal_df, background_colname):
    return bcorrected_signal_df[background_colname]


@pytest.fixture
def time_colname():
    return "time"


@pytest.fixture
def amp_colname():
    return "amp_raw"


@pytest.fixture
def mp():
    mp = MapPeaks()
    return mp


@pytest.fixture
def dp():
    dp = DataPrepper()
    return dp


@pytest.fixture
def int_col():
    return "amp_corrected"


@pytest.fixture
def psignals(
    chm: Chromatogram,
    time: FloatArray,
    stored_popt: DataFrame[Popt],
):
    psignals = chm._deconvolve._construct_peak_signals(time, stored_popt)

    return psignals


@pytest.fixture
def peak_report(
    dc: PeakDeconvolver,
    stored_popt: DataFrame[Popt],
    psignals: DataFrame[PSignals],
    timestep: np.float64,
) -> PReport:
    peak_report = dc._get_peak_report(
        stored_popt,
        psignals,
        timestep,
    )

    peak_report = PReport.validate(peak_report, lazy=True)

    return peak_report  # type: i


@pytest.fixture
def popt_parqpath():
    """
    Intended to be used to store a popt df as it is computationally expensive to deconvolute many-peaked windows
    """
    return "/Users/jonathan/hplc-py/tests/jonathan_tests/asschrompopt.parquet"


@pytest.fixture()
def stored_popt(popt_parqpath):
    """
    Read the stored popt_df, short circuiting the slow optimization process
    """
    return pd.read_parquet(popt_parqpath)


from hplc_py.deconvolve_peaks.mydeconvolution import DataPrepper, PeakDeconvolver

from hplc_py.map_signals.map_peaks import MapPeaks, PeakMap
from hplc_py.map_signals.map_windows import MapWindows, WindowedSignal


@pytest.fixture
def left_bases(
    pm: DataFrame[PeakMap],
) -> Series[pd.Int64Dtype]:
    left_bases: Series[pd.Int64Dtype] = Series[pd.Int64Dtype](
        pm[PeakMap.pb_left], dtype=pd.Int64Dtype()
    )
    return left_bases


@pytest.fixture
def right_bases(
    pm: DataFrame[PeakMap],
) -> Series[pd.Int64Dtype]:
    right_bases: Series[pd.Int64Dtype] = Series[pd.Int64Dtype](
        pm[PeakMap.pb_right], dtype=pd.Int64Dtype()
    )
    return right_bases


@pytest.fixture
def amp(
    amp_bcorr: FloatArray,
) -> Series[pd.Float64Dtype]:
    amp: Series[pd.Float64Dtype] = Series(
        amp_bcorr, name="amp", dtype=pd.Float64Dtype()
    )
    return amp


from hplc_py.map_signals.map_windows import MapWindows, WindowedSignal


@pytest.fixture
def mw() -> MapWindows:
    mw = MapWindows()
    return mw


@pytest.fixture
def pm(
    mp: MapPeaks,
    amp: Series[pd.Float64Dtype],
    time: Series[pd.Float64Dtype],
) -> DataFrame[PeakMap]:
    pm = mp.map_peaks(
        amp,
        time,
    )
    return pm


@pytest.fixture
def ws(
    mw: MapWindows,
    time: Series[pd.Float64Dtype],
    amp: Series[pd.Float64Dtype],
    left_bases: Series[pd.Float64Dtype],
    right_bases: Series[pd.Float64Dtype],
) -> DataFrame[WindowedSignal]:
    ws = mw.window_signal(
        left_bases,
        right_bases,
        time,
        amp,
    )
    return ws


@pytest.fixture
def main_chm_asschrom_loaded(
    in_signal: DataFrame,
):
    main_chm = hplc.quant.Chromatogram(
        pd.DataFrame(
            in_signal.rename(
                columns={
                    "amp_raw": "signal",
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
    return "/Users/jonathan/hplc-py/tests/jonathan_tests/main_chm_asschrom_fitted.pk"


@pytest.fixture
def main_chm_asschrom_fitted_pk(
    main_chm_asschrom_fitted_pkpth: str,
):
    import pickle

    return pickle.load(open(main_chm_asschrom_fitted_pkpth, "rb"))

@pytest.fixture
def main_params(main_chm_asschrom_fitted_pk):
    """
    Returns the main asschrom curve fit parameters in long form.

    columns: ['w_idx','p_idx','param','bng','val']
    """
    params = main_chm_asschrom_fitted_pk.params_jono

    params: pl.DataFrame = pl.from_pandas(params).melt(
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

@pytest.fixture
def main_scores(main_chm_asschrom_fitted_pk):
    scores = pl.LazyFrame(
        main_chm_asschrom_fitted_pk.scores.reset_index(drop=True)
    ).with_columns(window_id=pl.col("window_id") - 1)

    scores = scores.select([
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
            ])

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
        ).with_row_index("time_idx")

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
def main_windowed_peak_signals(
    main_window_props_windowed_signal: tuple[Any, Any]
):
    """
    The main package asschrom windowed input signal, not the reconstruction.

    columns: 'time_idx','w_idx','time_range','signal'
    """
    return main_window_props_windowed_signal[1]

@pytest.fixture
def main_window_df(main_chm_asschrom_fitted_pk: Any):
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
    )
    return window_df

@pytest.fixture
def main_unmixed_signals(main_chm_asschrom_fitted_pk: Any):
    unmixed_signals = (
        pl.DataFrame(main_chm_asschrom_fitted_pk.unmixed_chromatograms)
        .with_row_index("time_idx")
        .melt(id_vars="time_idx", value_name="amp_unmixed", variable_name="p_idx")
        .with_columns(
            p_idx=pl.col("p_idx").str.replace("column_", "").cast(pl.UInt32),
        )
    )
    return unmixed_signals

@pytest.fixture
def main_extract_popt_and_peak_window_recon_signal(
    main_chm_asschrom_fitted_pk: Any
):
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
                    time_idx=(pl.col("t_range") * 100).cast(pl.UInt32),
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
def main_popt(
    main_extract_popt_and_peak_window_recon_signal: tuple[Any, Any]
):
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

    Present with columns: ['time_idx','w_idx','p_idx','w_recon_sig']
    """

    return main_extract_popt_and_peak_window_recon_signal[1]

@pytest.fixture
def main_peak_map(
    main_chm_asschrom_fitted_pk
):
    """
    Main pakcage asschrom peak map
    
    The data gathered from scipy peak widths
    """
    peak_map = main_chm_asschrom_fitted_pk._peak_map_jono
    return peak_map

@pytest.fixture
def main_peak_widths_amp_input(
    main_chm_asschrom_fitted_pk
):
    """
    The amplitude series inputted into scipy peak widths by main. May or may not be the
    same as the output of `correct_baseline`, I have not confirmed that.
    """
    amp_input = main_chm_asschrom_fitted_pk._jono_assign_window_intensity
    return amp_input

@pytest.fixture
def __main_bcorr_interms_extract(
    main_chm_asschrom_fitted_pk,
):
    
    interms = main_chm_asschrom_fitted_pk._jono_bcorr_interms
    
    interm_signals = pl.DataFrame({k: v for k, v in interms.items() if k not in ['shift','n_iter']})
    interm_params = pl.DataFrame({k: v for k, v in interms.items() if k in ['shift','n_iter']})
    
    return interm_signals, interm_params
    
@pytest.fixture
def main_bcorr_interm_signals(
    __main_bcorr_interms_extract
):
    """
    Contains the intermediate calculations of the main baseline correctino for asschrom.
    
    columns: ['signal','tfrom_new','inv_tform','bcorr_not_rounded','bcorr_rounded']
    """
    return __main_bcorr_interms_extract[0]

@pytest.fixture
def main_bcorr_interm_params(
    __main_bcorr_interms_extract
):
    """
    contains the intermediate parameters calculated during main baseline correction of asschrom.
    
    dict keys: ['shift','n_iter']
    """
    return __main_bcorr_interms_extract[1]

