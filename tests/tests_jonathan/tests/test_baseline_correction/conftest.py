import pytest

from hplc_py.baseline_correction.baseline_correction import BaselineCorrection


@pytest.fixture
def background(bcorrected_signal_df, background_colname):
    return bcorrected_signal_df[background_colname]


@pytest.fixture
def windowsize():
    return 5


@pytest.fixture
def cb(
    n_iter: int,
):
    cb = BaselineCorrection(window_size=0.65, n_iter=n_iter, verbose=True)
    return cb


@pytest.fixture
def n_iter():
    return 250
