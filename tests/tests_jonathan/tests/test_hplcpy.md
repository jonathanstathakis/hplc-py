"""
Test module for testing the `HPLCPY` object, i.e. the main user interface for the pipeline and associated actions.
"""

import pytest
from hplc_py.hplcpy import HPLCPY
import numpy as np
from numpy.typing import NDArray
, int
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandera.typing import Series, DataFrame

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
        amp_raw: Series[float],
    )->HPLCPY:
        return HPLCPY(time.to_numpy(float), amp_raw.to_numpy(float))
    
    def test_plot_signal(
     self,
     hpy_loaded: HPLCPY,
     test_hpy_ax: Axes,
    )->None:
        hpy_loaded.plot_signal()
        
    def test_correct_baseline(
        self,
        hpy_loaded: HPLCPY,
    )->None:
        
        print(hpy_loaded.correct_baseline().chm)
        
    def test_map_peaks_no_correct(
        self,
        hpy_loaded: HPLCPY,
    ):
        hpy_loaded.map_peaks()

    def test_map_peaks_correct(
        self,
        hpy_loaded: HPLCPY,
    ):
        hpy_loaded.correct_baseline().map_peaks()
        
    def test_map_windows(
        self,
        hpy_loaded: HPLCPY,
    ):
        hpy_loaded.correct_baseline().map_windows()
        
        breakpoint()
      
    def test_deconvolve(
        self,
        hpy_loaded: HPLCPY,
        benchmark,
    ):
        benchmark(hpy_loaded.correct_baseline().map_windows().deconvolve)
    
    def test_fit_assess(
        self,
        hpy_loaded: HPLCPY,
    ):  
        hpy_loaded.correct_baseline().map_windows().deconvolve().assess_fit()
    
        
        
        
        
        