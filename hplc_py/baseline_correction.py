from numpy import float64
from pandera.typing import Series
import warnings
from typing import TypedDict, Optional
import numpy as np
import tqdm
from hplc_py.io_validation import IOValid
from numpy.typing import NDArray
from typing import Self


class BlineKwargs(TypedDict, total=False):
    windowsize: int
    verbose: bool


class CorrectBaseline(IOValid):

    def __init__(
        self,
        window_size: float = 5.0,
        verbose: Optional[bool] = True,
    ):

        self._window_size = window_size
        self.__verbose = verbose
        self.background: NDArray[float64] = np.ndarray(0)

    def fit(
        self,
        amp: NDArray[float64],
        timestep: float,
    ):
        """
        Fit the signal background using SNIP

        :param amp_raw: input raw signal with background to be fitted
        :type amp_raw: NDArray[float64]
        :param timestep: the average difference of the observation intervals in the time unit
        :type timestep: float
        :param windowsize: size of the filter window
        :type windowsize: float
        :param verbose: whether to report fit progress to console, defaults to False
        :type verbose: bool, optional
        :return: the fitted signal background
        :rtype: NDArray[float64]
        """
        amp = np.asarray(amp, float)
        if np.isnan(amp).any():
            raise ValueError("NaN detected in input amp")
        self.amp_raw = amp

        if np.isnan(timestep):
            raise ValueError("NaN detected as timestep input")

        shift = self._compute_shift(amp)
        amp_shifted = self._shift_amp(amp, shift)
        amp_shifted_clipped = self._clip_amp(amp_shifted)

        # compute the LLS operator to reduce signal dynamic range
        s_compressed = self._compute_compressed_signal(amp_shifted_clipped)

        # calculate the number of iterations for the minimization

        n_iter = self._compute_n_iter(self._window_size, timestep)

        # iteratively filter the compressed signal
        s_compressed_prime = self._compute_s_compressed_minimum(s_compressed, n_iter)

        # Perform the inverse of the LLS transformation and subtract

        inv_tform = self._compute_inv_tform(s_compressed_prime)

        background = np.add(inv_tform, shift)

        self.background = background

        return self

    def transform(
        self,
    ) -> Self:
        """
        Transform the input amplitude signal by subtracting the fitted background
        """
        self.corrected: Series[float] = Series[float](
            data=np.subtract(self.amp_raw, self.background), name="amp"
        ).rename_axis(index="t_idx")
        return self

    def _shift_amp(
        self,
        amp: NDArray[float64],
        shift: float,
    ) -> NDArray[float64]:

        amp_shifted = amp - shift
        return amp_shifted

    def _clip_amp(
        self,
        amp: NDArray[float64],
    ) -> NDArray[float64]:
        amp_ = np.asarray(amp, dtype=float)

        heaviside_sf: NDArray[float64] = np.heaviside(amp_, 0)

        amp_clipped: NDArray[float64] = amp_ * heaviside_sf
        
        return amp_clipped

    def _compute_n_iter(self, window_size: float, timestep: float) -> int:
        n_iter = int(np.divide(np.subtract(np.divide(window_size, timestep), 1), 2))
        return n_iter

    def _compute_compressed_signal(self, signal: NDArray[float64]) -> NDArray[float64]:
        """
        return a compressed signal using the LLS operator.
        """
        signal_ = np.asarray(signal, dtype=float)

        tform = np.log(np.log(np.sqrt(signal_ + 1) + 1) + 1)

        return tform

    def _compute_inv_tform(self, tform: NDArray[float64]) -> NDArray[float64]:
        # invert the transformer
        inv_tform = (np.exp(np.exp(tform) - 1) - 1) ** 2 - 1
        return inv_tform.astype(float)

    def _subtract_background(
        self,
        signal: NDArray[float64],
        inv_tform: NDArray[float64],
        shift: float,
    ) -> NDArray[float64]:
        transformed_signal = np.subtract(np.subtract(signal, shift), inv_tform)

        return transformed_signal.astype(float)

    def check_for_negatives(self, signal: NDArray[float64]) -> bool:
        has_negatives = False

        min_val = np.min(signal)
        max_val = np.max(signal)

        if np.less(min_val, 0):
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

    def _compute_shift(self, signal: NDArray[float64]) -> float:
        # the shift is computed as the median of the negative signal values
        signal_ = signal
        # signal_ = np.asarray(signal, dtype=float)

        has_negatives = self.check_for_negatives(signal_)

        if has_negatives:

            shift = np.median(signal_[signal_ < 0]).astype(float)
        else:
            shift = float(0.0)

        return shift

    def compute_iterator(self, n_iter: int) -> range:
        """
        return an iterator running from 1 to `n_iter`
        """
        return range(1, np.add(n_iter, 1))

    def _compute_s_compressed_minimum(
        self,
        s_compressed: NDArray[float64],
        n_iter: int,
    ) -> NDArray[float64]:
        """
        Apply the filter to find the minimum of s_compressed to approximate the baseline
        """
        # Iteratively filter the signal

        # set loading bar if verbose is True

        # Compute the number of iterations given the window size.

        _s_compressed = np.asarray(s_compressed, dtype=float)

        if _s_compressed.ndim != 1:
            raise ValueError(f"s_compressed must be 1D array, got {_s_compressed.ndim}")

        if self.__verbose:
            self._bg_correction_progress_state = 1
            iterator = tqdm.tqdm(
                self.compute_iterator(n_iter),
                desc="Performing baseline correction",
            )
        else:
            self._bg_correction_progress_state = 0
            iterator = self.compute_iterator(n_iter)  # type: ignore

        for i in iterator:
            s_compressed_prime = _s_compressed.copy()

            for j in range(i, len(_s_compressed) - i):
                s_compressed_prime[j] = min(
                    s_compressed_prime[j],
                    0.5 * (s_compressed_prime[j + i] + s_compressed_prime[j - i]),
                )

            _s_compressed = s_compressed_prime

        return _s_compressed
