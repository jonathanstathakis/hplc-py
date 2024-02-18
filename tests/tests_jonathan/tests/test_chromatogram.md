"""
Tests for the Chromatogram class
"""
import pytest
import numpy as np
from numpy import int64, float64
from numpy.typing import NDArray
from pandera.typing import Series, DataFrame

from hplc_py.chromatogram import Chromatogram
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from hplc_py.hplcpy import HPLCPY


class TestChromatogram:
    @pytest.fixture
    def test_chm_fig(self)->Figure:
        return plt.figure()

    @pytest.fixture
    def test_chm_ax(self, test_chm_fig: Figure)->Axes:
        return test_chm_fig.subplots()
    
    @pytest.fixture
    def int_array(self)->NDArray[int64]:
        return np.arange(0, 10, 1, dtype=int64)
    
    @pytest.fixture
    def float_array(self)->NDArray[float64]:
        return np.arange(0, 10, 1, dtype=float64)

    def test_chm_init(
        self,
        float_array: NDArray[float64]
    )->None:
        
        chm = Chromatogram(time=float_array, amp=float_array)
    
    @pytest.fixture
    def chm_loaded(
        self,
        time: Series[float64],
        amp_raw: Series[float64],
    )-> Chromatogram:
        
        chm = Chromatogram(time=time.to_numpy(float64), amp=amp_raw.to_numpy(float64))
        return chm
        
    def test_plot_signal(
        self,
        chm_loaded: Chromatogram,
        test_chm_ax: Axes,
    )->None:
        chm_loaded.plot_signals(test_chm_ax)
        
        plt.show()