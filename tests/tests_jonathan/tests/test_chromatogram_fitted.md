import pytest

from hplc_py.hplcpy import HPLCPY
from pandera.typing import Series

@pytest.fixture
def chm_fitted(
    time: Series[float],
    amp: Series[float],
):
    hpy = HPLCPY(time.to_numpy(float), amp.to_numpy(float))
    
    chm_fitted = hpy.fit_transform().chm
    
    return chm_fitted
    
def test_chm_fitted(
    chm_fitted,
):
    breakpoint()