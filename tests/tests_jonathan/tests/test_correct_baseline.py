"""
Test all aspects of the baseline correction module, which attempts to be as pure as possible, relying on numpy rather than higher level data structures, with the intention of porting to JAX at some point in the future.
"""
import pandera as pa
import numpy as np
import pytest
from numpy import float64
from numpy.typing import NDArray
from pandera.typing.pandas import DataFrame, Series

from hplc_py.baseline_correct.correct_baseline import CorrectBaseline
from hplc_py.hplc_py_typing.hplc_py_typing import RawData

def test_s_compressed_prime_exec(s_compressed_prime: NDArray[float64]):
    pass

def test_correct_baseline(
    amp_raw: Series[float64],
    amp_bcorr: Series[float64],
):
    """
    Tests if the average amplitude in amp_bcorr is less than the average amplitude
    of amp_raw. Assumes that baseline correction works in both the positive and negative
    domains, which I beleive as I have formulated it, it does not.
    """
    assert amp_raw.abs().mean() > amp_bcorr.abs().mean()




@pytest.fixture
def windowsize():
    return 5

@pytest.fixture
def shift(
    amp_raw,
    cb: CorrectBaseline,
) -> float64:
    shift = cb._compute_shift(amp_raw)
    return shift


@pytest.fixture
def amp_shifted_clipped(
    cb: CorrectBaseline,
    amp_raw: NDArray[float64],
    shift: float64,
) -> NDArray[float64]:
    amp_shifted = cb._shift_amp(amp_raw, shift)

    amp_shifted_clipped = cb._clip_amp(amp_shifted)
    return amp_shifted_clipped


@pytest.fixture
def s_compressed(
    cb: CorrectBaseline, amp_shifted_clipped: NDArray[float64]
) -> NDArray[float64]:
    # intensity raw compressed
    s_compressed = cb._compute_compressed_signal(amp_shifted_clipped)

    return s_compressed


def test_amp_compressed_exists_and_is_array(
    s_compressed: NDArray[float64],
):
    assert np.all(s_compressed)
    assert isinstance(s_compressed, np.ndarray)


@pytest.fixture
def n_iter(cb: CorrectBaseline, windowsize: int, timestep: float64):
    n_iter = cb._compute_n_iter(windowsize, timestep)

    return n_iter


@pytest.fixture
def s_compressed_prime(
    cb: CorrectBaseline,
    s_compressed: NDArray[float64],
    n_iter: int,
):
    s_compressed_prime = cb._compute_s_compressed_minimum(
        s_compressed,
        n_iter,
    )
    return s_compressed_prime
