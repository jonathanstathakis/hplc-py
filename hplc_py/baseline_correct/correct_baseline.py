import warnings
from dataclasses import dataclass, field
from typing import TypedDict, cast

import numpy as np
import pandas as pd
import pandera as pa
import tqdm
from pandera.typing import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import FloatArray
from hplc_py.misc.misc import LoadData, TimeStep


class BlineKwargs(TypedDict, total=False):
    windowsize: int
    verbose: bool


class SignalDFBCorr(pa.DataFrameModel):
    time_idx: pd.Int64Dtype
    time: pd.Float64Dtype
    amp_raw: pd.Float64Dtype
    amp_corrected: pd.Float64Dtype
    background: pd.Float64Dtype

    class Config:
        strict = True


@dataclass
class CorrectBaseline(TimeStep, LoadData):
    _bg_corrected = False
    _int_corr_col_suffix = "_corrected"
    _windowsize = (None,)
    _n_iter = None
    _verbose = True
    _background_col = "background"
    _bg_correction_progress_state = None
    _debug_bcorr_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def correct_baseline(
        self,
        windowsize: int = 5,
        verbose: bool = False,
        precision=9,
    ) -> DataFrame[SignalDFBCorr]:
        """
        Correct baseline and add corrected amplitude and background series to signal_df
        """
        if not any(cast(pd.DataFrame, self._signal_df)):
            raise RuntimeError("Run `set_signal_df` first")

        if not self._timestep:
            self.set_timestep(self._signal_df[self.time_col])

        if not isinstance(self._signal_df, pd.DataFrame):
            raise TypeError(
                f"signal_df must be pd.DataFrame, got {type(self._signal_df)}"
            )

        if not isinstance(self.amp_col, str):
            raise TypeError("amp_col must be a string")
        if self.amp_col not in self._signal_df.columns:
            raise ValueError(f"amp_col: {self.amp_col} not in signal_df")

        timestep = np.float64(self._timestep)

        if not isinstance(windowsize, int):
            raise TypeError("windowsize must be int")

        shift = np.float64(0)

        self._verbose = verbose
        self._precision = precision

        if self._bg_corrected:
            warnings.warn(
                "Baseline has already been corrected. Rerunning on original signal..."
            )

        # extract the amplitude series
        amp_raw = self._signal_df.loc[:, self.amp_col].to_numpy(np.float64)

        # enforce array datatype
        amp_raw = np.asarray(amp_raw, np.float64)

        if amp_raw.ndim != 1:
            raise ValueError("amp has too many dimensions, please enter a 1D array")

        has_negatives = self.check_for_negatives(amp_raw)

        if has_negatives:
            # Clip the signal if the median value is negative
            shift = self.compute_shift(amp_raw)
        else:
            shift=0
        
        # dont know why we do this    
        amp_shifted, shift = self.shift_amp(amp_raw, shift)
        
        # dont know why we do this
        amp_shifted_clipped = self.clip_amp(amp_shifted)

        # compute the LLS operator to reduce signal dynamic range
        s_compressed = self.compute_compressed_signal(amp_shifted_clipped)

        # calculate the number of iterations for the minimization

        n_iter = self.compute_n_iter(windowsize, timestep)

        # iteratively filter the compressed signal
        s_compressed_prime = self.compute_s_compressed_minimum(
            s_compressed, n_iter, verbose
        )

        # Perform the inverse of the LLS transformation and subtract

        inv_tform = self.compute_inv_tform(s_compressed_prime)

        background = self.compute_background(inv_tform, shift)
        
        amp_bcorr = amp_raw - background

        self._bg_corrected = True

        len_debug_bcorr_df = len(amp_raw)

        self._debug_bcorr_df = pd.DataFrame(
            {
                "amp_raw": pd.Series(amp_raw),
                "timestep": pd.Series([timestep] * len_debug_bcorr_df),
                "shift": pd.Series([shift] * len_debug_bcorr_df),
                "n_iter": pd.Series([n_iter] * len_debug_bcorr_df),
                "s_compressed": pd.Series(s_compressed),
                "s_compressed_prime": pd.Series(s_compressed_prime),
                "inv_tform": pd.Series(inv_tform),
                "y_corrected": pd.Series(amp_bcorr),
                "background": pd.Series(background),
            }
        )

        self.amp_col = self.amp_col.replace("raw", "corrected")

        self._signal_df[self.amp_col] = pd.Series(amp_bcorr, dtype=pd.Float64Dtype())

        self._signal_df["background"] = pd.Series(background, dtype=pd.Float64Dtype())

        return self._signal_df        
    
    def shift_amp(
        self,
        amp: FloatArray,
        shift,
    ):
        amp_shifted = amp - shift
        return amp_shifted, shift
    
    def clip_amp(
        self,
        amp,
    ):

        heaviside_sf = np.heaviside(amp, 0)

        amp_clipped = amp * heaviside_sf
        return amp_clipped
        

    def compute_n_iter(self, window_size, timestep):
        return int(((window_size / timestep) - 1) / 2)

    def compute_compressed_signal(self, signal: FloatArray) -> FloatArray:
        """
        return a compressed signal using the LLS operator.
        """
        tform = np.log(np.log(np.sqrt(signal + 1) + 1) + 1)

        return tform.astype(np.float64)

    def compute_inv_tform(self, tform: FloatArray) -> FloatArray:
        # invert the transformer
        inv_tform = (np.exp(np.exp(tform) - 1) - 1) ** 2 - 1
        return inv_tform.astype(np.float64)

    def _subtract_background(
        self,
        signal: FloatArray,
        inv_tform: FloatArray,
        shift: np.float64,
    ) -> FloatArray:
        transformed_signal = signal -shift - inv_tform

        return transformed_signal.astype(np.float64)

    def check_for_negatives(self, signal: FloatArray) -> bool:
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

    def compute_shift(self, signal: FloatArray) -> np.float64:
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
        s_compressed: FloatArray,
        n_iter: int,
        verbose: bool = True,
    ) -> FloatArray:
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
                    0.5 * (s_compressed_prime[j + i] + s_compressed_prime[j - i]),
                )

            _s_compressed = s_compressed_prime

        return _s_compressed

    def compute_background(
        self,
        inv_tform: FloatArray,
        shift: np.float64,
    ) -> FloatArray:
        background = inv_tform + shift

        return background
