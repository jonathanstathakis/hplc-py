
import numpy as np
from numpy.typing import NDArray
from numpy import float64, int64
from .chromatogram import Chromatogram
from .show import Show
from typing import Self, Optional
from .baseline_correct.correct_baseline import CorrectBaseline

class HPLCPY:
    """
    The main pipeline class containing a number of optional methods in the form of classed submodules: baseline correction, windowing, deconvolution, fit assessment, visualisation.

    Methods can be chained to produce pipelines, with the outcome of the pipelines stored in the chromatogram object. Post pipeline, the chromatogram object can be extracted for further processing or comparison.
    """
    def __init__(self, time: NDArray[float64], amp: NDArray[float64]):    
        self.chm = Chromatogram(time, amp)
        self.show = Show()
        
    def correct_baseline(
        self,
        n_iter: Optional[int] = 0,
        _bg_corrected = False,
        _corrected_suffix: str = "_corrected",
        _windowsize: tuple = (None,),
        _verbose: bool = True,
        _background_col: str = "background",
    ):
        CorrectBaseline(
            corrected_suffix=_corrected_suffix,
            windowsize=_windowsize,
            verbose=_verbose,
            background_colname=_background_col,
                        ).fit_transform()