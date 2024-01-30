"""
Test module for testing the `HPLCPY` object, i.e. the main user interface for the pipeline and associated actions.
"""

import pytest
from hplc_py.hplcpy import HPLCPY
import numpy as np
from numpy.typing import NDArray
from numpy import float64, int64
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

class TestHPLCPY:
    @pytest.fixture
    def test_hpt_fig(self)->Figure:
        return plt.figure()

    @pytest.fixture
    def test_hpy_ax(self, test_hpt_fig: Figure)->Axes:
        return test_hpt_fig.subplots()
    
    @pytest.fixture
    def hpy_loaded(
        self,
        time:NDArray[float64],
        amp: NDArray[float64],
    )->HPLCPY:
        return HPLCPY(time, amp)
    
    def test_plot_signal(
     self,
     hpy_loaded: HPLCPY,
     test_hpy_ax: Axes,
    )->None:
        hpy_loaded.plot_signal()
        
        