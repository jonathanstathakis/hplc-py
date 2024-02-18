import pytest
from numpy import float64
from hplc_py.hplcpy import HPLCPY
from pandera.typing import Series

@pytest.fixture
def chm_fitted(
    time: Series[float64],
    amp: Series[float64],
):
    hpy = HPLCPY(time.to_numpy(float64), amp.to_numpy(float64))
    
    chm_fitted = hpy.fit_transform().chm
    
    return chm_fitted
    
def test_chm_fitted(
    chm_fitted,
):
    breakpoint()