"""
Flow:

1. assess_fit
2. compute reconstruction score
3. apply tolerance
4. calculate mean fano
6. Determine whether window reconstruction score and fano ratio is acceptable
7. create 'report card'
8. 

Notes:
- reconstruction score defined as unmixed_signal_AUC+1 divided by mixed_signal_AUC+1. 

"""

import pandas as pd

import numpy as np
import numpy.typing as npt

import pandera as pa
import pandera.typing as pt

from hplc_py.hplc_py_typing.hplc_py_typing import OutPeakReportBase
class FitAsessment:    
    
    def _score_df_factory(
        self,
        mixed_signal_df,
        unmixed_signal_df,
        peak_report_df,
    ):
    
        
        """
        per window peak window, calculate:
        - [x] signal_var: variance of window signal
        - [x] signal_mean: mean of window signal
        - [x] signal fano = signal_var / signal_mean and store it as a var.
        - [x] window area: abs of sum of amp in window + 1
        - [x] window peaks: peak indexes
        - [x] window_peak_area: area of recon signal of each peak
        - [x] score: divide window_peak_area by window_area
        - [ ] 
        """
        
        
        
    
    def compute_signal_var(
        self,
        signal: npt.NDArray[np.float64],
    ):
        signal_var = np.var(signal)
        
        return signal_var
    
    def compute_signal_mean(
        self,
        signal: npt.NDArray[np.float64],
    ):
        signal_mean = np.mean(signal)
        
        return signal_mean
        
    def compute_fano(
        self,
        signal_var: npt.NDArray[np.float64],
        signal_mean: npt.NDArray[np.float64],
    ):
        
        fano = signal_var / signal_mean

        return fano
    
    def compute_mixed_auc(
        self,
        signal: npt.NDArray[np.float64],
    ):
        mixed_auc = signal.sum()+1

        return mixed_auc

    
    def get_peak_idxs(
        self,
        window_idx: int,
        peak_report_df: pt.DataFrame[OutPeakReportBase],
    ):
        
        peak_idxs = peak_report_df.query("window_idx==@window_idx")['peak_idx'].to_numpy(np.int64)
        
        return peak_idxs
    
    def compute_unmixed_auc(
        self,
        window_idx: int,
        peak_report_df: pt.DataFrame[OutPeakReportBase],
    ):
        
        unmixed_auc = (peak_report_df.query("window_idx==@window_idx")['unmixed_area'].sum()+1).to_numpy(np.float64)
        
        return unmixed_auc
    
    def compute_score(
        self,
        mixed_auc: np.float64,
        unmixed_auc: np.float64,
    ):
        score = unmixed_auc/mixed_auc
        
        return score