import numpy as np

import pandas as pd
import pandera as pa
import pandera.typing as pt
import tqdm
import warnings
import copy
from typing import Any
import numpy.typing as npt

import matplotlib.pyplot as plt


class BaselineCorrector:
    def __init__(self):
        self._bg_corrected = False
        self._int_corr_col_suffix = "_corrected"
        self._windowsize = (None,)
        self._n_iter = None
        self._verbose = True
        self._background_col = "background"
        self._bg_correction_progress_state = None
        self._debug_bcorr_df=pd.DataFrame()
    
    def correct_baseline(
        self,
        amp_raw: npt.NDArray[np.float64],
        timestep: np.float64,
        windowsize: int = 5,
        verbose: bool = False,
        precision=9,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        R""" """

        shift = np.float64(0)
        self._verbose = verbose
        self._precision = precision

        if self._bg_corrected:
            warnings.warn(
                "Baseline has already been corrected. Rerunning on original signal..."
            )

        # enforce array datatype
        amp_raw = np.asarray(amp_raw, np.float64)

        if amp_raw.ndim != 1:
            raise ValueError("amp has too many dimensions, please enter a 1D array")


        amp_clipped = self.shift_and_clip_amp(amp_raw)

        # compute the LLS operator to reduce signal dynamic range
        s_compressed = self.compute_compressed_signal(amp_clipped)

        # calculate the number of iterations for the minimization
        
        n_iter = self.compute_n_iter(windowsize, timestep)
        
        # iteratively filter the compressed signal
        s_compressed_prime = self.compute_s_compressed_minimum(
            s_compressed, n_iter, verbose
        )

        # Perform the inverse of the LLS transformation and subtract

        inv_tform = self.compute_inv_tform(s_compressed_prime)

        amp_bcorr = self.transform_and_subtract(
            amp_raw, inv_tform, shift
        )

        background = self.compute_background(inv_tform, shift)

        self._bg_corrected = True
        
        len_debug_bcorr_df=len(amp_raw)
        
        self._debug_bcorr_df=pd.DataFrame(
            {
                'amp_raw':pd.Series(amp_raw),
                'timestep':pd.Series([timestep]*len_debug_bcorr_df),
                'shift':pd.Series([shift]*len_debug_bcorr_df),
                'n_iter':pd.Series([n_iter]*len_debug_bcorr_df),
                's_compressed':pd.Series(s_compressed),
                's_compressed_prime':pd.Series(s_compressed_prime),
                'inv_tform':pd.Series(inv_tform),
                'y_corrected':pd.Series(amp_bcorr),
                'background':pd.Series(background),
        })
                             
        return amp_bcorr, background
    
    def shift_and_clip_amp(
        self,
        amp: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # Look at the relative magnitudes of the maximum and minimum values
        # And raise a warning if there are appreciable negative peaks.
        has_negatives = self.check_for_negatives(amp)

        if has_negatives:
            # Clip the signal if the median value is negative
            shift = self.compute_shift(amp)
        else:
            shift = 0
            
        amp_shifted = amp - shift
        
        heaviside_sf = np.heaviside(amp_shifted, 0)
        
        amp_clipped = amp_shifted * heaviside_sf
        return amp_clipped
    
    def compute_n_iter(
        self,
        window_size,
        timestep
    ):
        return int(((window_size / timestep) - 1) / 2)
    
    def compute_compressed_signal(
        self, signal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        return a compressed signal using the LLS operator.
        """
        tform = np.log(np.log(np.sqrt(signal + 1) + 1) + 1)

        return tform.astype(np.float64)

    def compute_inv_tform(
        self, tform: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # invert the transformer
        inv_tform = (np.exp(np.exp(tform) - 1) - 1) ** 2 - 1
        return inv_tform.astype(np.float64)

    def transform_and_subtract(
        self,
        signal: npt.NDArray[np.float64],
        inv_tform: npt.NDArray[np.float64],
        shift: np.float64,
    ) -> npt.NDArray[np.float64]:
        
        transformed_signal = signal - inv_tform# - shift

        return transformed_signal.astype(np.float64)

    def check_for_negatives(self, signal: npt.NDArray[np.float64]) -> bool:
        has_negatives = False

        min_val = np.min(signal)
        max_val = np.max(signal)

        if min_val < 0:
            has_negatives = True

            # check for ratio of negative to positive values, if greater than 10% warn user
            if (np.abs(min_val) / max_val) >= 0.1:
                warnings.warn(
                    """
                \x1b[30m\x1b[43m\x1b[1m
                The chromatogram appears to have appreciable negative signal . Automated background 
                subtraction may not work as expected. Proceed with caution and visually 
                check if the subtraction is acceptable!
                \x1b[0m"""
                )

        return has_negatives

    def compute_shift(self, signal: npt.NDArray[np.float64]) -> np.float64:
        # the shift is computed as the median of the negative signal values

        shift = np.median(signal[signal < 0])

        shift = shift.astype(np.float64)

        return shift

    def compute_iterator(self, n_iter: int):
        """
        return an iterator running from 1 to `n_iter`
        """
        return range(1, n_iter + 1)

    def compute_s_compressed_minimum(
        self,
        s_compressed: npt.NDArray[np.float64],
        n_iter: int,
        verbose: bool = True,
    ) -> npt.NDArray[np.float64]:
        """
        Apply the filter to find the minimum of s_compressed to approximate the baseline
        """
        # Iteratively filter the signal

        # set loading bar if verbose is True

        # Compute the number of iterations given the window size.
    

        _s_compressed = np.asarray(s_compressed, dtype=np.float64)

        if _s_compressed.ndim != 1:
            raise ValueError(f"s_compressed must be 1D array, got {s_compressed.ndim}")

        if verbose:
            self._bg_correction_progress_state = 1
            iterator = tqdm.tqdm(
                self.compute_iterator(n_iter),
                desc="Performing baseline correction",
            )
        else:
            self._bg_correction_progress_state = 0
            iterator = self.compute_iterator(n_iter)

        for i in iterator:
            s_compressed_prime = _s_compressed.copy()

            for j in range(i, len(_s_compressed) - i):
                s_compressed_prime[j] = min(
                    s_compressed_prime[j],
                    0.5 * (s_compressed_prime[j + i] + s_compressed_prime[j - i]))
                
            _s_compressed = s_compressed_prime

        return _s_compressed

    def compute_background(
        self,
        inv_tform: npt.NDArray[np.float64],
        shift: np.float64 = np.float64(0),
    ) -> npt.NDArray[np.float64]:
        background = inv_tform + shift

        return background
