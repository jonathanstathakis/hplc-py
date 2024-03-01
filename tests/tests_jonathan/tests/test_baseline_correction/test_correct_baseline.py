"""
Test all aspects of the baseline correction module, which attempts to be as pure as possible, relying on numpy rather than higher level data structures, with the intention of porting to JAX at some point in the future.
"""

import pytest
import hvplot
from numpy import float64
from numpy.typing import NDArray
from pandera.typing.pandas import Series

from hplc_py.baseline_correction.baseline_correction import BaselineCorrection


@pytest.fixture
def n_iter() -> int:
    return 250


def test_s_compressed_prime_exec(s_compressed_prime: NDArray[float64]):
    pass


def test_correct_baseline(
    amp_raw: Series[float],
    amp_bcorr: Series[float],
    n_iter: int,
):
    """
    Tests if the average amplitude in amp_bcorr is less than the average amplitude
    of amp_raw. Assumes that baseline correction works in both the positive and negative
    domains, which I beleive as I have formulated it, it does not.
    """
    assert amp_raw.abs().mean() > amp_bcorr.abs().mean()


def test_correct_baseline_viz(amp_raw: Series[float]):

    bc = BaselineCorrection(n_iter=250, window_size=1, verbose=True)
    bc.fit(X=amp_raw)
    signals = bc.correct_baseline()
    viz = bc.viz_baseline_correction()

    hvplot.show(viz)
