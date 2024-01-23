from hplc_py.hplc_py_typing.hplc_py_typing import SignalDFInBase
from hplc_py.quant import Chromatogram


import pandera as pa
import pandera.typing as pt
import pytest


from typing import Any


class TestFitPeaks:
    """
    test the `fit_peaks` call, which performs the overall process to unmix the peaks and provide a peak table
    """

    @pytest.fixture
    def fit_peaks(
        self,
        chm: Chromatogram,
        in_signal: pt.DataFrame[SignalDFInBase],
    ):
        chm.set_signal_df(in_signal)

        popt_df, unmixed_df = chm.fit_peaks()

        return popt_df, unmixed_df

    @pytest.fixture
    def popt_df(self, fit_peaks: tuple[Any, Any]):
        return fit_peaks[0]

    @pytest.fixture
    def unmixed_df(self, fit_peaks: tuple[Any, Any]):
        return fit_peaks[1]

    @pa.check_types
    def test_fit_peaks_exec(
        self,
        fitted_chm: Chromatogram,
    ):
        assert fitted_chm