import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pt
import tqdm
import warnings
import copy
from typing import Any
import numpy.typing as npt

class BaselineCorrector:
    
    def __init__(self):
        self._bg_corrected = False
        self._int_corr_col_suffix = "_corrected"
        self._windowsize=None,
        self._n_iter=None
        self._verbose=True
        self._background_col = 'background'
        
        self._bg_correction_progress_state=None
        
    def correct_baseline(self,
                         signal: npt.NDArray[np.float64],
                         windowsize:int=5,
                         timestep:np.float64=np.float64(0),
                         verbose:bool=True,
                         precision=9)->tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        R"""
        Performs Sensitive Nonlinear Iterative Peak (SNIP) clipping to estimate 
        and subtract background in chromatogram.

        Parameters
        ----------
        window : `int`
            The approximate size of signal objects in the chromatogram in dimensions
            of time. This is related to the number of iterations undertaken by 
            the SNIP algorithm.
        return_df : `bool`
            If `True`, then chromatograms (before and after background correction) are returned
        verbose: `bool`
            If `True`, progress will be printed to screen as a progress bar. 
        precision: `int`
            The number of decimals to round the subtracted signal to. Default is 9.

        Returns
        -------
        corrected_df : `pandas.core.frame.DataFrame`
            If `return_df = True`, then the original and the corrected chromatogram are returned.

        Notes
        -----
        This implements the SNIP algorithm as presented and summarized in `Morhác
        and Matousek 2008 <https://doi.org/10.1366/000370208783412762>`_. The 
        implementation here also rounds to 9 decimal places in the subtracted signal
        to avoid small values very near zero.
        """

        shift = np.float64(0)
        self._verbose=verbose
        self._precision=precision
        
        if self._bg_corrected:
            warnings.warn(
                'Baseline has already been corrected. Rerunning on original signal...')
        
        
        # enforce array datatype
        amp = np.asarray(signal, np.float64)
        
        if amp.ndim!=1:
            raise ValueError("amp has too many dimensions, please enter a 1D array")

        # Look at the relative magnitudes of the maximum and minimum values
        # And throw a warning if there are appreciable negative peaks.
        
        has_negatives = self.check_for_negatives(amp)
        
        if has_negatives:
            # Clip the signal if the median value is negative
            shift = self.compute_shift(amp)
            
            amp = self.clip_signal(amp, shift)
        
        # compute the LLS operator to reduce signal dynamic range
        s_compressed = self.compute_compressed_signal(amp)
        
        # iteratively filter the compressed signal
        s_compressed_prime = self.compute_s_compressed_minimum(s_compressed, windowsize, timestep, verbose)

        # Perform the inverse of the LLS transformation and subtract
        
        inv_tform = self.compute_inv_tform(s_compressed_prime)
        
        background_corrected_intensity =self.transform_and_subtract(amp,
                                                                    precision,
                                                                    inv_tform,
                                                                    shift
                                                                    )
        
        background =self.compute_background(inv_tform, shift)
        
        self._bg_corrected = True
        
        return background_corrected_intensity, background
        
        

    def compute_compressed_signal(self, signal: npt.NDArray[np.float64])-> npt.NDArray[np.float64]:
        """
        return a compressed signal using the LLS operator.
        """
        tform = np.log(np.log(np.sqrt(signal + 1) + 1) + 1)
        
        return tform.astype(np.float64)
    
    def compute_inv_tform(self, tform: npt.NDArray[np.float64])-> npt.NDArray[np.float64]:
        # invert the transformer
        inv_tform = ((np.exp(np.exp(tform) - 1) - 1)**2 - 1)
        return inv_tform.astype(np.float64)
        
    def transform_and_subtract(self,
                               signal: npt.NDArray[np.float64],
                               precision:int,
                               inv_tform: npt.NDArray[np.float64],
                               shift:np.float64|int=0,
                               )->npt.NDArray[np.float64]:
        
        
        transformed_signal = np.round((signal - shift - inv_tform), decimals=precision)
        
        return transformed_signal.astype(np.float64)

    def check_for_negatives(self, signal:npt.NDArray[np.float64])->bool:
        
        has_negatives = False
        
        min_val = np.min(signal)
        max_val = np.max(signal)
        
        if min_val < 0:
            
            has_negatives = True
            
            # check for ratio of negative to positive values, if greater than 10% warn user
            if (np.abs(min_val) / max_val) >= 0.1:
                
                warnings.warn("""
                \x1b[30m\x1b[43m\x1b[1m
                The chromatogram appears to have appreciable negative signal . Automated background 
                subtraction may not work as expected. Proceed with caution and visually 
                check if the subtraction is acceptable!
                \x1b[0m""")

        return has_negatives

    def compute_shift(self, signal: npt.NDArray[np.float64])->np.float64:

        # the shift is computed as the median of the negative signal values
            
        shift = np.median(signal[signal < 0])
        
        shift = shift.astype(np.float64)
        
        return shift
    
    def clip_signal(self, signal: npt.NDArray[np.float64], shift: np.float64)-> npt.NDArray[np.float64]:
        
        signal -= shift
        
        signal *= np.heaviside(signal, 0)
        
        return signal.astype(np.float64)
    
    def compute_n_iter(self, windowsize: int, dt: float)->int:
        
        assert isinstance(windowsize, int)
        assert isinstance(dt, float)
        n_iter = int(((windowsize / dt) - 1) / 2)
        return n_iter

    def compute_iterator(self, n_iter:int):
        """
        return an iterator running from 1 to `n_iter`
        """
        return range(1, n_iter + 1)
    
    def compute_s_compressed_minimum(self, s_compressed: npt.NDArray[np.float64], windowsize:int, timestep: int|float, verbose:bool=True) -> npt.NDArray[np.float64]:
        """
        Apply the filter to find the minimum of s_compressed to approximate the baseline
        """
        # Iteratively filter the signal
        
        # set loading bar if verbose is True
        
        # Compute the number of iterations given the window size.
        self._n_iter=self.compute_n_iter(windowsize, timestep)
        
        if verbose:
            self._bg_correction_progress_state = 1
            iterator = tqdm.tqdm(self.compute_iterator(self._n_iter),
                            desc="Performing baseline correction")
        else:
            self._bg_correction_progress_state = 0
            iterator = self.compute_iterator(self._n_iter)
        
        # avoid Unbound warning
        s_compressed_prime = pd.Series([])
        
        for i in iterator:
            s_compressed_prime = s_compressed.copy()
            
            for j in range(i, len(s_compressed) - i):
                
                s_compressed_prime[j] = min(s_compressed_prime[j],
                                0.5 * (s_compressed_prime[j+i] + s_compressed_prime[j-i]))
        
        return s_compressed_prime.astype(np.float64)
    
    def compute_background(self,
                           inv_tform: npt.NDArray[np.float64],
                           shift:np.float64=np.float64(0),
                           )->npt.NDArray[np.float64]:
        
        background = inv_tform + shift
        
        return background