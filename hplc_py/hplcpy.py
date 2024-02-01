import numpy as np
from numpy.typing import NDArray
from numpy import float64, int64
from .chromatogram import Chromatogram
from .show import Show
from typing import Self, Optional
from .baseline_correct.correct_baseline import CorrectBaseline
from .map_signals.map_peaks.map_peaks import MapPeaks
from .map_signals.map_windows import MapWindows
from .deconvolve_peaks.mydeconvolution import PeakDeconvolver
from hplc_py.io_validation import IOValid

class HPLCPY(IOValid):
    """
    The main pipeline class containing a number of optional methods in the form of classed submodules: baseline correction, windowing, deconvolution, fit assessment, visualisation.

    Methods can be chained to produce pipelines, with the outcome of the pipelines stored in the chromatogram object. Post pipeline, the chromatogram object can be extracted for further processing or comparison.
    """

    def __init__(self, time: NDArray[float64], amp: NDArray[float64]):
        self._chm = Chromatogram(time, amp)
        self.show = Show()

    @property
    def chm(self):
        return self._chm
    
    @chm.getter
    def chm(self):
        return self._chm
    
    def deconv_pipeline(
        self,
    ):
        """
        Apply all the contained methods to baseline correct, deconvolve and assess the fit.
        """

    def correct_baseline(
        self,
        verbose: bool = True,
        windowsize: int = 5,
    ):
        cb = CorrectBaseline(
            windowsize=windowsize,
            verbose=verbose,
        )

        amp = self.chm._data["amp"].to_numpy(float64)
        timestep = self.chm.timestep

        bcorr = (
            cb.fit(
                amp=amp,
                timestep=timestep,
            )
            .transform()
            .corrected
        )

        background = cb.background

        self.chm._data["amp" + "_corrected"] = bcorr
        self.chm._data["background"] = background

        self.chm.bcorr_corrected = True

        return self

    def map_peaks(
        self,
    ) -> Self:
        
        amp = self.chm.amp
        time = self.chm._data.time
        timestep = self.chm.timestep
        
        pm = MapPeaks().map_peaks(amp=amp, time=time, timestep=timestep)

        self.chm.peakmap = pm

        return Self

    def map_windows(self) -> Self:
        """
        First map the peaks then windows, cant do one without the other. if the peaks
        are already mapped, skip that step
        """
        if not self.is_nonempty_df(self.chm.peakmap):
            self.map_peaks()

        left_bases = self.chm.peakmap["pb_left_idx"]
        right_bases = self.chm.peakmap["pb_right_idx"]
        time = self.chm.time
        amp = self.chm.amp
        
        wm = MapWindows().window_signal(
            left_bases=left_bases, right_bases=right_bases, time=time, amp=amp
        )

        w_time_idx = wm[["w_type", "w_idx", "time_idx"]]
        
        self.chm.join_data_to_windowed_time(w_time_idx)
        
        return self

    def deconvolve(self) -> Self:
        """
        Deconvolve based on windows assigned in map_windows
        """
        pm = self.chm.peakmap
        ws = self.chm.ws
        timestep = self.chm.timestep
        pd = PeakDeconvolver(pm=pm, ws=ws, timestep=timestep)

        pd.deconvolve_peaks()

        self.chm.popt = pd.popt_df
        self.chm.psignals = pd.psignals
        self.chm._data = pd.ws
        self.chm.preport = pd.preport

        return self

    def assess_fit(self) -> Self:
        from .fit_assessment import FitAssessment

        ws = self.chm.ws
        scores = FitAssessment().assess_fit(ws=ws)

        self.chm.scores = scores

        return self

    def fit_transform(self) -> Self:
        """
        Implement the pipeline from end to end
        """
        self.correct_baseline().map_windows().deconvolve()

        return self
