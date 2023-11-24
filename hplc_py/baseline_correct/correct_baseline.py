import numpy as np
import pandas as pd
import tqdm
import warnings

class BaselineCorrector:
    def correct_baseline(self,
                         window=5,
                         return_df=False,
                         verbose=True,
                         precision=9):
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
        if self._bg_corrected == True:
            warnings.warn(
                'Baseline has already been corrected. Rerunning on original signal...')
            self.int_col = self.int_col.split('_corrected')[0]

        # Unpack and copy dataframe and intensity profile
        df = self.df
        signal = df[self.int_col].copy()

        # Look at the relative magnitudes of the maximum and minimum values
        # And throw a warning if there are appreciable negative peaks.
        signal, shift = self.check_for_negatives(signal)

        # Compute the LLS operator
        tform = np.log(np.log(np.sqrt(signal.values + 1) + 1) + 1)

        # Compute the number of iterations given the window size.
        n_iter = int(((window / self._dt) - 1) / 2)

        # Iteratively filter the signal
        if verbose:
            self._bg_correction_progress_state = 1
            iter = tqdm.tqdm(range(1, n_iter + 1),
                             desc="Performing baseline correction")
        else:
            self._bg_correction_progress_state = 0
            iter = range(1, n_iter + 1)

        for i in iter:
            tform_new = tform.copy()
            for j in range(i, len(tform) - i):
                tform_new[j] = min(tform_new[j], 0.5 *
                                   (tform_new[j+i] + tform_new[j-i]))
            tform = tform_new

        # Perform the inverse of the LLS transformation and subtract
        self.df = self.transform_and_subtract(precision, df, shift, tform)
        
        self._bg_corrected = True
        
        self.int_col = f'{self.int_col}_corrected'
        
        if return_df:
            return df

    def transform_and_subtract(self, precision, df, shift, tform):
        inv_tform = ((np.exp(np.exp(tform) - 1) - 1)**2 - 1)
        
        df[f'{self.int_col}_corrected'] = np.round(
            (df[self.int_col].values - shift - inv_tform), decimals=precision)
        
        df[f'estimated_background'] = inv_tform + shift
        
        return df

    def check_for_negatives(self, signal):
        min_val = np.min(signal)
        max_val = np.max(signal)
        if min_val < 0:
            if (np.abs(min_val) / max_val) >= 0.1:
                warnings.warn("""
\x1b[30m\x1b[43m\x1b[1m
The chromatogram appears to have appreciable negative signal . Automated background 
subtraction may not work as expected. Proceed with caution and visually 
check if the subtraction is acceptable!
\x1b[0m""")

        # Clip the signal if the median value is negative
        if (signal < 0).any():
            shift = np.median(signal[signal < 0])
        else:
            shift = 0
        signal -= shift
        signal *= np.heaviside(signal, 0)
        return signal, shift