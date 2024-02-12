import warnings
from typing import TypedDict, Optional
import numpy as np
import tqdm
from hplc_py.io_validation import IOValid
from numpy.typing import NDArray
from numpy import float64
from typing import Self


class BlineKwargs(TypedDict, total=False):
    windowsize: int
    verbose: bool


class CorrectBaseline(IOValid):

    def __init__(
        self,
        window_size: float = 5.0,
        verbose: bool=True,
    ):

        self.background: NDArray[float64] = np.ndarray(0)
        
        self._window_size = window_size

        self._amp = NDArray[float64] = np.ndarray(0)
        self._timestep: float = 0.0
        
        self._verbose = verbose

    def fit(
        self,
        amp: NDArray[float64],
        timestep: float,
        n_iter: int,
    ):
        """
        Fit the signal background using SNIP

        :param amp_raw: input raw signal with background to be fitted
        :type amp_raw: NDArray[float64]
        :param timestep: the average difference of the observation intervals in the time unit
        :type timestep: float64
        :param windowsize: size of the filter window
        :type windowsize: float
        :param verbose: whether to report fit progress to console, defaults to False
        :type verbose: bool, optional
        :return: the fitted signal background
        :rtype: NDArray[float64]
        """
        if np.isnan(amp).any():
            raise ValueError("NaN detected in input amp")
        self._amp = amp

        if np.isnan(timestep):
            raise ValueError("NaN detected as timestep input")

        self._timestep = timestep

        self.n_iter = n_iter

        return self

    def transform(
        self,
    ) -> Self:
        """
        Transform the input amplitude signal by subtracting the fitted background
        """
        self.background = compute_background(self._amp, self.n_iter, self._verbose)

        return self


def compute_background(amp: NDArray[float64], n_iter: int, verbose: bool = True):

    shift = compute_shift(amp)

    amp_shifted = shift_amp(amp, shift)

    amp_shifted_clipped = clip_amp(amp_shifted)

    # compute the LLS operator to reduce signal dynamic range
    s_compressed = compute_compressed_signal(amp_shifted_clipped)

    # iteratively filter the compressed signal
    s_compressed_prime = compute_s_compressed_minimum(
        s_compressed, n_iter, verbose=verbose
    )

    # Perform the inverse of the LLS transformation and subtract

    inv_tform = compute_inv_tform(s_compressed_prime)

    background = np.add(inv_tform, shift)
    return background


def shift_amp(
    amp: NDArray[float64],
    shift: float64,
) -> NDArray[float64]:

    amp_shifted = amp - shift
    return amp_shifted


def clip_amp(
    amp: NDArray[float64],
) -> NDArray[float64]:
    amp_ = np.asarray(amp, dtype=float64)

    heaviside_sf = np.heaviside(amp_, 0)

    amp_clipped = amp_ * heaviside_sf
    return amp_clipped


def compute_n_iter(window_size: float, timestep: float) -> int:

    return int(np.divide(np.subtract(np.divide(window_size, timestep), 1), 2))


def compute_compressed_signal(signal: NDArray[float64]) -> NDArray[float64]:
    """
    return a compressed signal using the LLS operator.
    """
    signal_ = np.asarray(signal, dtype=float64)

    tform = np.log(np.log(np.sqrt(signal_ + 1) + 1) + 1)

    return tform


def compute_inv_tform(tform: NDArray[float64]) -> NDArray[float64]:
    # invert the transformer
    inv_tform = (np.exp(np.exp(tform) - 1) - 1) ** 2 - 1
    return inv_tform.astype(float64)


def subtract_background(
    signal: NDArray[float64],
    inv_tform: NDArray[float64],
    shift: float64,
) -> NDArray[float64]:
    transformed_signal = np.subtract(np.subtract(signal, shift), inv_tform)

    return transformed_signal.astype(float64)


def check_for_negatives(signal: NDArray[float64]) -> bool:
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


def compute_shift(signal: NDArray[float64]) -> float64:
    # the shift is computed as the median of the negative signal values
    signal_ = signal
    # signal_ = np.asarray(signal, dtype=float64)

    has_negatives = check_for_negatives(signal_)

    if has_negatives:

        shift = np.median(signal_[signal_ < 0]).astype(float64)
    else:
        shift = float64(0.0)

    return shift


def compute_iterator(n_iter: int) -> range:
    """
    return an iterator running from 1 to `n_iter`
    """
    return range(1, np.add(n_iter, 1))


def compute_s_compressed_minimum(
    s_compressed: NDArray[float64],
    n_iter: int,
    verbose=True,
) -> NDArray[float64]:
    """
    Apply the filter to find the minimum of s_compressed to approximate the baseline
    """
    # Iteratively filter the signal

    # set loading bar if verbose is True

    # Compute the number of iterations given the window size.

    _s_compressed = np.asarray(s_compressed, dtype=float64)

    if _s_compressed.ndim != 1:
        raise ValueError(f"s_compressed must be 1D array, got {_s_compressed.ndim}")

    if verbose:
        iterator = tqdm.tqdm(
            compute_iterator(n_iter),
            desc="Performing baseline correction",
        )
    else:
        iterator = compute_iterator(n_iter)  # type: ignore

    for i in iterator:
        s_compressed_prime = _s_compressed.copy()

        for j in range(i, len(_s_compressed) - i):
            s_compressed_prime[j] = min(
                s_compressed_prime[j],
                0.5 * (s_compressed_prime[j + i] + s_compressed_prime[j - i]),
            )

        _s_compressed = s_compressed_prime

    return _s_compressed
