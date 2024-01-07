import os

import pytest
import pandera as pa
import pandera.typing as pt
import pandas as pd

import numpy as np
import numpy.typing as npt

from hplc_py.hplc_py_typing.hplc_py_typing import (
    SignalDFInBase,
    OutSignalDF_Base,
    SignalDFInAssChrom,
    OutPeakDF_Base,
    OutPeakReportBase,
    OutWindowDF_Base,
    OutInitialGuessBase,
    OutDefaultBoundsBase,
    OutReconDFBase,
    OutPoptDF_Base,
    
    
    isArrayLike,
)

from hplc_py.quant import Chromatogram
import json

class AssChromResults:
    """
    The results of 'fit_peaks' on the 'test_assessment_chrom' dataset

    to add:

    - [x] timestep. Has been added as a column of 'param_df'

    """

    def __init__(self):
        
        '''
        load results tables from the main project from parquet files whose paths are
        given as a json file. store the tables as a member dict.
        '''
        
        
        in_basepath = "/Users/jonathan/hplc-py/tests/jonathan_tests/main_asschrom_results"
        paths_json_inpath = os.path.join(in_basepath, 'filepaths.json')
        
        with open(paths_json_inpath, 'r') as f:
        
            self.paths_dict = json.load(f)
        
        self.tables = {}
        
        # keys: ['peaks', 'unmixed', 'asschrom_param_tbl', 'mixed_signals', 'bcorr_dbug_tbl', 'window_df']
        for name, path in self.paths_dict.items():
            self.tables[name]=pd.read_parquet(path)
        self.tables['adapted_param_tbl'] = self.adapt_param_df(self.tables['asschrom_param_tbl'])
        
        
            
        return None

    def adapt_param_df(self, param_df):
        """
        adapt main param_df to be expressed as per the definition that `popt_factory` is expecting

        """
        from scipy.optimize._lsq.common import in_bounds

        self.timestep = self.tables['asschrom_param_tbl'].loc[0, "timestep"]
        
        # 
        df: pd.DataFrame = param_df.copy(deep=True)
        
        # assign inbounds column
        df['inbounds']=df.apply(lambda x: in_bounds(x["p0"], x["lb"], x["ub"]),axis=1) #type: ignore
        
        # replace parameter values to match my labels
        df["param"] = df['param'].replace(
                    {
                        "amplitude": "amp",
                        "location": "loc",
                        "scale": "whh",
                    }
                )
        
        # set param column as ordered categorical
        df['param'] = pd.Categorical(values=df["param"], categories=df["param"].unique(), ordered=True)
        
        # provide an ascending peak idx col irrespective of windows
        df['peak_idx'] = np.repeat([0,1,2,3],len(df)/4)
        
        # remove the timestep col
        df = df.drop(['timestep'],axis=1)
        
        return df

    def get_results_paths(self):
        out_basepath = (
            "/Users/jonathan/hplc-py/tests/jonathan_tests/main_asschrom_results"
        )

        ptable_outpath = os.path.join(out_basepath, "asschrom_peak_table.parquet")
        unmixed_outpath = os.path.join(out_basepath, "asschrom_unmixed.parquet")
        param_df_outpath = os.path.join(out_basepath, "param_df.parquet")
        mixed_signals_outpath = os.path.join(
            out_basepath, "asschrom_mixedsignals.parquet"
        )

        return {
            "param_outpath": param_df_outpath,
            "ptable_outpath": ptable_outpath,
            "unmixed_outpath": unmixed_outpath,
            "mixed_outpath": mixed_signals_outpath,
        }

@pytest.fixture
def acr():
    acr = AssChromResults()
    return acr

        
@pytest.fixture
def amp_raw_main(
    acr: AssChromResults,
):
    amp_raw = acr.tables['mixed_signals']['y']
    
    return amp_raw 


@pytest.fixture
def target_window_df(acr):
    return acr.tables['peaks']

@pytest.fixture
def datapath():
    return "tests/test_data/test_assessment_chrom.csv"


@pytest.fixture
@pa.check_types
def in_signal(datapath: str) -> pt.DataFrame[SignalDFInBase]:
    data = pd.read_csv(datapath)

    data = data.rename({"x": "time", "y": "amp_raw"}, axis=1, errors='raise')
    data.insert(0, "tbl_name", "testsignal")

    return data


@pa.check_types
def test_in_signal_matches_schema(in_signal: pt.DataFrame[SignalDFInBase]) -> None:
    'currently in signal is asschrom'
    SignalDFInAssChrom(in_signal)
    return None


@pytest.fixture
def chm():
    return Chromatogram()


@pytest.fixture
def time(in_signal: pt.DataFrame[SignalDFInBase]):
    assert isinstance(in_signal, pd.DataFrame)
    assert isArrayLike(in_signal.time)
    return in_signal.time.values


@pytest.fixture
def timestep(chm: Chromatogram, time: npt.NDArray[np.float64]) -> float:
    timestep = chm.compute_timestep(time)
    assert timestep
    assert isinstance(timestep, float)
    assert timestep > 0

    return timestep


@pytest.fixture
def amp_raw(in_signal: pt.DataFrame[SignalDFInBase]):
    amp = in_signal['amp_raw'].values

    return amp


@pytest.fixture
def windowsize():
    return 5


@pytest.fixture
def bcorr_col(intcol: str) -> str:
    return intcol + "_corrected"


@pytest.fixture
def timecol():
    return "time"


@pytest.fixture
def intcol():
    return "signal"


@pytest.fixture
def bcorr_tuple(
    chm: Chromatogram,
    amp_raw: npt.NDArray[np.float64],
    timestep: np.float64,
    windowsize: int,
):
    y_corrected, background = chm._baseline.correct_baseline(
        amp_raw,
        timestep,
        windowsize,
        verbose=False,
    )

    return y_corrected, background


@pytest.fixture
def amp_bcorr(bcorr_tuple):
    return bcorr_tuple[0]


@pytest.fixture
def background(bcorr_tuple):
    return bcorr_tuple[1]


@pytest.fixture
def amp_cn(
    chm: Chromatogram, amp_bcorr: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    ``int_cn` has the base data as the first element of the namespace then the process
    in order. i.e. intensity: [corrected, normalized]
    """
    int_cn = chm._findwindows.normalize_series(amp_bcorr)

    assert any(int_cn)
    assert isArrayLike(int_cn)
    assert np.min(int_cn >= 0)
    assert np.max(int_cn <= 1)

    return int_cn


@pytest.fixture
def peak_df(
    chm: Chromatogram,
    time: npt.NDArray[np.float64],
    amp_bcorr: npt.NDArray[np.float64],
) -> pt.DataFrame[OutPeakDF_Base]:
    peak_df = chm._findwindows.peak_df_factory(time, amp_bcorr)

    return peak_df


@pytest.fixture
def norm_amp(
    chm: Chromatogram,
    amp_bcorr: npt.NDArray[np.float64],
):
    norm_int = chm._findwindows.normalize_series(amp_bcorr)
    return norm_int


@pytest.fixture
def time_colname():
    return "time"


@pytest.fixture
def amp_colname():
    return "amp_raw"

@pytest.fixture
def signal_df(time,
              amp_raw,
              amp_bcorr,
              background
              ):
    return pd.DataFrame({
        'time_idx':np.arange(0, len(time),1),
        'time':time,
        'amp_raw': amp_raw,
        'amp_corrected': amp_bcorr,
        'amp_bg': background
    })

@pytest.fixture
def window_df(
    chm: Chromatogram,
    signal_df: pt.DataFrame,
    peak_df: pt.DataFrame[OutPeakDF_Base],
)->pt.DataFrame:
    window_df = chm._findwindows.window_df_factory(
        signal_df['time'].to_numpy(np.float64),
        signal_df['amp_corrected'].to_numpy(np.float64),
        peak_df['rl_left'].to_numpy(np.float64),
        peak_df['rl_right'].to_numpy(np.float64)
    )

    return window_df

@pytest.fixture
@pa.check_types
def p0_df(
    chm: Chromatogram,
    signal_df: pt.DataFrame[OutSignalDF_Base],
    peak_df: pt.DataFrame[OutPeakDF_Base],
    window_df: pt.DataFrame[OutWindowDF_Base],
    timestep: np.float64,
    int_col: str,
) -> pt.DataFrame[OutInitialGuessBase]:
    p0_df = chm._deconvolve.dataprepper.p0_factory(
        signal_df,
        peak_df,
        window_df,
        timestep,
        int_col,
    )
    return p0_df


@pytest.fixture
@pa.check_types
def default_bounds(
    chm: Chromatogram,
    p0_df: pt.DataFrame[OutInitialGuessBase],
    signal_df: pt.DataFrame[OutSignalDF_Base],
    window_df: pt.DataFrame[OutWindowDF_Base],
    peak_df: pt.DataFrame[OutPeakDF_Base],
    timestep: np.float64,
)->pt.DataFrame[OutDefaultBoundsBase]:
    default_bounds = chm._deconvolve.dataprepper.default_bounds_factory(
        p0_df,
        signal_df,
        window_df,
        peak_df,
        timestep,
    )
    return default_bounds

@pa.check_types
def test_default_bounds_tbl_init(
    default_bounds: pt.DataFrame[OutDefaultBoundsBase],
):
    'use check_types to test the input tbl'
    pass
@pytest.fixture
def int_col():
        return 'amp_corrected'
    


@pytest.fixture
def xdata(
    signal_df,
):
    return signal_df["time"]


@pytest.fixture
def unmixed_df(
    chm: Chromatogram,
    xdata,
    stored_popt,
):
    unmixed_df = chm._deconvolve._reconstruct_peak_signal(xdata, stored_popt)
    unmixed_df = unmixed_df.reset_index(names='time_idx')
    unmixed_df = unmixed_df.loc[:,['peak_idx','time_idx','time','unmixed_amp']]
    return unmixed_df


@pytest.fixture
def peak_report(
    chm: Chromatogram,
    stored_popt: OutPoptDF_Base,
    unmixed_df: OutReconDFBase,
    timestep: np.float64,
) -> OutPeakReportBase:
    peak_report = chm._deconvolve.compile_peak_report(
        stored_popt,
        unmixed_df,
        timestep,
    )
    return peak_report.pipe(pt.DataFrame[OutPeakReportBase])  # type: ignore

@pytest.fixture
def windowed_signal_df(
    chm: Chromatogram,
    signal_df,
    window_df,
):
    windowed_signal = chm._deconvolve.dataprepper._window_signal_df(signal_df,window_df)    
    
    return windowed_signal


@pytest.fixture
def my_param_df(
    chm: Chromatogram,
    p0_df,
    default_bounds,
):
    params = chm._deconvolve.dataprepper._param_df_factory(p0_df,default_bounds,)
    
    return params

@pytest.fixture
def popt_df(
    chm: Chromatogram,
    windowed_signal_df,
    my_param_df,
):
    popt_df = chm._deconvolve._popt_factory(
        windowed_signal_df,
        my_param_df,    
    )
    return popt_df

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
def fitted_chm(
    chm: Chromatogram,
    amp_colname: str,
    time_colname: str,
    signal_df: pt.DataFrame[SignalDFInBase],
    
):
    chm.load_data(signal_df)

    chm.fit_peaks(
        amp_colname,
        time_colname, 
    )
    
    return chm