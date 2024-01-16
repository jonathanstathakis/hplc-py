"""
2023-11-27 08:56:43

Design notes:

Chromatogram object will have a df member, class objects such as BaselineCorrector will not, only provide methods that operate on that df. therefore for testing will have to invert chm to apply on df.

2023-12-10 18:09:25

Response to above:

that concept worked for baseline corrector, which operates on a 1 dimensional array of data with length N, but
for the peak deconvolution etc operating on an indexed array is better for development purposes, if not the most computationally efficient method.

"""
import typing
from dataclasses import dataclass, field

import pandas as pd
import pandera as pa
import pandera.typing as pt
import numpy as np
import numpy.typing as npt

import scipy.signal
import scipy.optimize
import scipy.special
import tqdm
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import termcolor

from .misc.misc import LoadData

from hplc_py.baseline_correct.correct_baseline import CorrectBaseline, BlineKwargs
from hplc_py.map_signals.map_peaks import MapPeaks
from hplc_py.map_signals.map_windows import MapWindows

from hplc_py.deconvolve_peaks import mydeconvolution
from hplc_py.fit_assessment import FitAssessment
from hplc_py import show 
from hplc_py.hplc_py_typing.hplc_py_typing import (
    SignalDF,
    Recon,
    OutPeakReportBase,
)

from typing import TypedDict



@dataclass
class Chromatogram(LoadData):
  
    # initialize
    _correct_bline:bool=True
    _viz:bool=True
    
    # member class objects
    _baseline=CorrectBaseline()
    _ms=MapWindows()
    _deconvolve=mydeconvolution.PeakDeconvolver()
    _fitassess=FitAssessment()
    _show=show.Show()
    
    # keys for the time and amp columns
    time_col: str = ""
    amp_col:str = ""
    
    _crop_offset = 0
    window_props = ""
    scores = ""
    _peak_indices = ""
    
    # to store the list of WindowState classes prior to deconvolve peaks DEBUGGING
    windowstates:list = field(default_factory=list)
    

    
    @pa.check_types
    def fit_peaks(
        self,
        correct_baseline: bool=True,
        bcorr_kwargs: BlineKwargs={},
        fwindows_kwargs: dict={},
        deconvolve_kwargs: dict={},
        verbose: bool=True,
    )->tuple[pt.DataFrame[OutPeakReportBase], pt.DataFrame[Recon]]:
        '''
        Process master method
        '''
        
        # baseline correction
        
        timestep = self._timestep
        
        # test if supplied column names are in the df
        for k, v in {'amp_colname':self.amp_col, 'time_colname':self.time_col}.items():
            
            if v not in self._signal_df.columns.to_list():
                raise ValueError(f"{k} value: {v} not in signal_df. Possible values are: {self._signal_df.columns}")
            
        time = self._signal_df.loc[:,self.time_col].to_numpy(np.float64)
        amp_raw = self._signal_df.loc[:,self.amp_col].to_numpy(np.float64)
        
        # baseline correction
        
        signal_df = signal_df.pipe(self._baseline.correct_baseline, timestep, **bcorr_kwargs, verbose=verbose)
        
        self._signal_df["amp_corrected"] = bcorr
        self._signal_df["amp_bg"] = background
        
        # peak profiling and windowing
        
        p_df, w_df = self._ms.profile_peaks_assign_windows(
            time,
            bcorr,
            **fwindows_kwargs,
        )
        
        self.peak_df = p_df
        self.window_df = w_df
        
        # peak deconvolution
        
        self.popt_df, selSignalDFelf._deconvolve.deconvolve_peaks(
            pt.DataFrame[OutSignalDF_Base](self._signal_df),
            self.peak_df,
            self.window_df,
            timestep
        )
        
        self.peak_report = self._deconvolve._get_peak_report(
            self.popt_df,
            self.unmixed_df,
            timestep,
        )
        return self.peak_report, self.unmixed_df