from hplc_py import AMPCORR
from hplc_py.baseline_correct.correct_baseline import CorrectBaseline
from hplc_py.hplc_py_typing.hplc_py_typing import FloatArray
from hplc_py.quant import Chromatogram


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import float64
from numpy.typing import NDArray
from pandera.typing.pandas import DataFrame
from scipy import integrate

from typing import Any, Literal


class TestCorrectBaseline:
    @pytest.fixture
    def windowsize(self):
        return 5

    @pytest.fixture
    def amp_clipped(
        self,
        cb: CorrectBaseline,
        amp_raw: FloatArray,
    ) -> FloatArray:
        return cb.shift_and_clip_amp(amp_raw)

    @pytest.fixture
    def s_compressed(self, cb: CorrectBaseline, amp_clipped: FloatArray) -> FloatArray:
        # intensity raw compressed
        s_compressed = cb.compute_compressed_signal(amp_clipped)

        return s_compressed

    def test_amp_compressed_exists_and_is_array(
        self,
        s_compressed: NDArray[float64],
    ):
        assert np.all(s_compressed)
        assert isinstance(s_compressed, np.ndarray)

    @pytest.fixture
    def n_iter(
        self,
        chm: Chromatogram,
        windowsize: Literal[5],
        timestep: float,
    ):
        return chm._baseline.compute_n_iter(windowsize, timestep)

    @pytest.fixture
    def s_compressed_prime(
        self,
        cb: CorrectBaseline,
        s_compressed: NDArray[float64],
        n_iter: int,
    ):
        s_compressed_prime = cb.compute_s_compressed_minimum(
            s_compressed,
            n_iter,
        )
        return s_compressed_prime

    def test_s_compressed_prime_exec(self, s_compressed_prime: FloatArray):
        pass

    def test_compute_inv_tfrom(
        self,
        chm: Chromatogram,
        amp_raw: FloatArray,
    ) -> None:
        chm._baseline.compute_inv_tform(amp_raw)

    def test_correct_baseline(
        self,
        loaded_cb: CorrectBaseline,
        time_colname: str,
        amp_raw: FloatArray,
    ) -> None:
        # pass the test if the area under the corrected signal is less than the area under the raw signal

        signal_df = loaded_cb.correct_baseline()

        amp_bcorr = signal_df[AMPCORR].to_numpy(np.float64)
        time = signal_df[time_colname].to_numpy(np.float64)

        x_start = time[0]
        x_end = time[-1]
        n_x = len(time)

        # add a preset baseline to ensure that correction behavior is within expected scale

        x = np.linspace(x_start, x_end, n_x)

        from scipy import stats

        skew = 1
        loc = x_end * 0.3
        scale = x_end * 0.3
        skewnorm = stats.skewnorm(skew, loc=loc, scale=scale)

        y = skewnorm.pdf(x) * np.power(np.max(amp_raw), 2)  # type: ignore

        added_baseline = amp_raw + y

        baseline_auc = integrate.trapezoid(added_baseline, time)

        bcorr_auc = integrate.trapezoid(amp_bcorr, time)

        assert baseline_auc > bcorr_auc

    def test_baseline_compare_main(
        self,
        amp_bcorr,
        main_window_df,
    ):
        """
        Compare the differences in baseline correction between the main and my approach
        """
        import polars as pl
        import hvplot
        from holoviews.plotting import bokeh

        bokeh.ElementPlot.width = 10000
        bokeh.ElementPlot.height = 10000

        df = pl.DataFrame(
            {
                "main": main_window_df["signal_corrected"],
                "mine": amp_bcorr,
                "amp_raw": main_window_df["signal"],
            }
        ).with_columns(
            main_my_diff=pl.col("main") - pl.col("mine"),
            my_raw_diff=pl.col("amp_raw") - pl.col("mine"),
            main_raw_diff=pl.col("amp_raw") - pl.col("main"),
        )

        breakpoint()

    def test_main_interms(
        self,
        main_bcorr_interm_params,
        main_bcorr_interm_signals
    ):
        breakpoint()
        pass