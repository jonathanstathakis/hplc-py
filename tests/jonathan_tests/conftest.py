import os
from typing import Any, Literal

import hplc
import pandas as pd
import polars as pl
import pytest
from numpy import float64, int64
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series

from hplc_py.baseline_correct.correct_baseline import CorrectBaseline, SignalDFBCorr
from hplc_py.deconvolve_peaks.mydeconvolution import DataPrepper, PeakDeconvolver
from hplc_py.hplc_py_typing.custom_checks import col_a_less_than_col_b  # noqa: F401
from hplc_py.hplc_py_typing.hplc_py_typing import (
    Popt,
    PReport,
    PSignals,
    SignalDFLoaded,
    WindowedSignal,
    RSignal
)
from hplc_py.io_validation import (
    IOValid
)
from hplc_py.map_signals.map_peaks.map_peaks import MapPeaks, PeakMap
from hplc_py.map_signals.map_windows import MapWindows
from hplc_py.misc.misc import compute_timestep


@pytest.fixture
def datapath():
    return "tests/test_data/test_assessment_chrom.csv"


@pytest.fixture
def in_signal(
    datapath: str,
    amp_col: str,
    time_col: str,
    ) -> DataFrame[SignalDFLoaded]:
    data = pd.read_csv(datapath)

    data = data.rename({"x": time_col, "y": amp_col}, axis=1, errors="raise").reset_index(names='time_idx').rename_axis(index='idx')
    data = DataFrame[SignalDFLoaded](data)
    return data


@pytest.fixture
def time(in_signal: DataFrame[SignalDFLoaded])->Series[float64]:
    return Series[float64](in_signal["time"])


@pytest.fixture
def timestep(time: NDArray[float64]) -> float:
    timestep: float = compute_timestep(time)
    return timestep


@pytest.fixture
def amp_raw(
    in_signal: DataFrame[SignalDFLoaded],
    amp_col: str,
    )->Series[float64]:
    
    amp = Series[float64](in_signal[amp_col])

    return amp


@pytest.fixture
def windowsize():
    return 5


@pytest.fixture
def bcorr_colname(amp_col: str) -> str:
    bcorr_col_str: str = amp_col+'_corrected'
    return bcorr_col_str


@pytest.fixture
def time_col():
    return "time"


@pytest.fixture
def amp_col():
    return "amp"

@pytest.fixture
def cb():
    cb = CorrectBaseline()
    return cb


@pytest.fixture
def bcorred_signal_df_asschrom(
    cb: CorrectBaseline,
    in_signal: DataFrame[SignalDFLoaded],
    amp_col: str,
    timestep: float,
    windowsize: int,
    ) -> DataFrame[SignalDFBCorr]:
    
    bcorred_signal_df = cb.fit_transform(
        signal_df=in_signal,
        amp_col=amp_col,
        timestep=timestep,
        windowsize=windowsize,
    )

    return bcorred_signal_df


@pytest.fixture
def amp_bcorr(bcorred_signal_df_asschrom: DataFrame, bcorr_colname: str)->Series[float64]:
    return Series[float64](bcorred_signal_df_asschrom[bcorr_colname])


@pytest.fixture
def background_colname():
    return "background"


@pytest.fixture
def background(bcorrected_signal_df, background_colname):
    return bcorrected_signal_df[background_colname]


@pytest.fixture
def mp():
    mp = MapPeaks()
    return mp


@pytest.fixture
def dp():
    dp = DataPrepper()
    return dp

@pytest.fixture
def dc() -> PeakDeconvolver:
    dc = PeakDeconvolver()
    return dc

@pytest.fixture
def int_col():
    return "amp_corrected"


@pytest.fixture
def time_pd_series(
    time: NDArray[float64],
)->Series[float64]:
    return Series[float64](pd.Series(time, name='time'))

@pytest.fixture
def psignals(
    dc: PeakDeconvolver,
    time_pd_series: Series[float64],
    stored_popt: DataFrame[Popt],
):
    psignals = dc._construct_peak_signals(time_pd_series, stored_popt)

    return psignals


@pytest.fixture
def peak_report(
    dc: PeakDeconvolver,
    stored_popt: DataFrame[Popt],
    psignals: DataFrame[PSignals],
    timestep: float64,
) -> DataFrame[PReport]:
    peak_report = dc._get_peak_report(
        stored_popt,
        psignals,
        timestep,
    )
 
    return peak_report

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



@pytest.fixture
def left_bases(
    my_peak_map: DataFrame[PeakMap],
) -> Series[int64]:
    left_bases: Series[int64] = Series[int64](
        my_peak_map[PeakMap.pb_left], dtype=int64
    )
    return left_bases


@pytest.fixture
def right_bases(
    my_peak_map: DataFrame[PeakMap],
) -> Series[int64]:
    right_bases: Series[int64] = Series[int64](
        my_peak_map[PeakMap.pb_right], dtype=int64
    )
    return right_bases


@pytest.fixture
def mw() -> MapWindows:
    mw = MapWindows()
    return mw

@pytest.fixture
def prom()->float:
    return 0.01

@pytest.fixture
def my_peak_map(
    mp: MapPeaks,
    amp_bcorr: Series[float64],
    prom: float,
    timestep: float,
    time: Series[float64],
) -> DataFrame[PeakMap]:
    pm = mp.map_peaks(
        amp=amp_bcorr,
        time=time,
        timestep=timestep,
        prominence=prom,
        wlen=None,
        find_peaks_kwargs={}
    )
    return pm


@pytest.fixture
def ws(
    mw: MapWindows,
    time_pd_series: Series[float64],
    amp_bcorr: Series[float64],
    left_bases: Series[float64],
    right_bases: Series[float64],
) -> DataFrame[WindowedSignal]:
    
    check_input_is_pd_series_float64(time_pd_series)    
    check_input_is_pd_series_float64(amp_bcorr)    
    check_input_is_pd_series_int64(left_bases)    
    check_input_is_pd_series_int64(right_bases)    
    
    
    ws = mw.window_signal(
        left_bases,
        right_bases,
        time_pd_series,
        amp_bcorr,
    )
    
    return ws


@pytest.fixture
def main_chm_asschrom_loaded(
    in_signal: DataFrame,
    amp_col: str,
):
    main_chm = hplc.quant.Chromatogram(
        pd.DataFrame(
            in_signal.rename(
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
    pkpth = "/Users/jonathan/hplc-py/tests/jonathan_tests/main_chm_asschrom_fitted.pk"
    
    return pkpth


@pytest.fixture
def main_chm_asschrom_fitted_pk(
    main_chm_asschrom_fitted_pkpth: str,
)->hplc.quant.Chromatogram:
    if not os.path.isfile("/Users/jonathan/hplc-py/tests/jonathan_tests/main_chm_asschrom_fitted.pk"):
        raise RuntimeError("No main pickle file found. ./tests/jonathan_tests/test_main_asschrom.py::test_pk_main_chm_asschrom_fitted must be run first.")
      
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
    """
    Main asschrom windowed signal - input and corrected, with background.
    
    Columns: ['window_type', 'window_id', 'time_idx', 'time', 'signal', 'signal_corrected', 'estimated_background']
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
def main_pm_(
    main_chm_asschrom_fitted_pk
):
    """
    Main package asschrom peak map
    
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

@pytest.fixture
def r_signal(
    dc: PeakDeconvolver,
    psignals: DataFrame[PSignals],
)-> DataFrame[RSignal]:
    
    r_signal = dc._reconstruct_signal(psignals)
    
    return r_signal