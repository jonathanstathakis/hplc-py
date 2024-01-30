"""
Test all aspects of the baseline correction module, which attempts to be as pure as possible, relying on numpy rather than higher level data structures, with the intention of porting to JAX at some point in the future.
"""

import numpy as np
import pytest
from numpy import float64
from numpy.typing import NDArray
from pandera.typing.pandas import DataFrame

from hplc_py.baseline_correct.correct_baseline import CorrectBaseline, SignalDFBCorr


class TestCorrectBaseline:
    @pytest.fixture
    def windowsize(self):
        return 5

    @pytest.fixture
    def shift(
        self,
        amp_raw,
        cb: CorrectBaseline,
    ) -> float64:
        shift = cb._compute_shift(amp_raw)
        return shift

    @pytest.fixture
    def amp_shifted_clipped(
        self,
        cb: CorrectBaseline,
        amp_raw: NDArray[float64],
        shift: float64,
    ) -> NDArray[float64]:
        amp_shifted = cb._shift_amp(amp_raw, shift)

        amp_shifted_clipped = cb._clip_amp(amp_shifted)
        return amp_shifted_clipped

    @pytest.fixture
    def s_compressed(
        self, cb: CorrectBaseline, amp_shifted_clipped: NDArray[float64]
    ) -> NDArray[float64]:
        # intensity raw compressed
        s_compressed = cb._compute_compressed_signal(amp_shifted_clipped)

        return s_compressed

    def test_amp_compressed_exists_and_is_array(
        self,
        s_compressed: NDArray[float64],
    ):
        assert np.all(s_compressed)
        assert isinstance(s_compressed, np.ndarray)

    @pytest.fixture
    def n_iter(self, cb: CorrectBaseline, windowsize: int, timestep: float64):
        n_iter = cb._compute_n_iter(windowsize, timestep)

        return n_iter

    @pytest.fixture
    def s_compressed_prime(
        self,
        cb: CorrectBaseline,
        s_compressed: NDArray[float64],
        n_iter: int,
    ):
        s_compressed_prime = cb._compute_s_compressed_minimum(
            s_compressed,
            n_iter,
        )
        return s_compressed_prime

    def test_s_compressed_prime_exec(self, s_compressed_prime: NDArray[float64]):
        pass

    def test_correct_baseline_asschrom(
        self,
        amp_bcorr: DataFrame[SignalDFBCorr],
    ):
        SignalDFBCorr.validate(amp_bcorr)
        pass

    def test_baseline_compare_main(
        self,
        amp_bcorr,
        main_window_df,
    ):
        """
        Compare the differences in baseline correction between the main and my approach
        """
        import polars as pl
        from holoviews.plotting import bokeh

        bokeh.ElementPlot.width = 10000
        bokeh.ElementPlot.height = 10000

        df = (
            pl.DataFrame(
                {
                    "main": main_window_df["signal_corrected"],
                    "mine": amp_bcorr,
                    "amp_raw": main_window_df["signal"],
                }
            )
            .with_columns(mine=pl.col('mine'))
            .with_columns(
                main_my_diff=(pl.col("main") - pl.col("mine")).abs(),
                diff_tol=pl.lit(0.05),
            )
            .with_columns(
                main_my_diff_prc=\
                    pl.when(pl.col('main_my_diff').ne(0)
                      )
                      .then(
                          pl.col('main_my_diff')
                          .truediv(pl.col('mine').abs())
                          )
                      .otherwise(0)
            )
            .with_columns(diffpass=pl.col("main_my_diff") < pl.col("diff_tol"))
        )
        breakpoint()
        assert df.filter('diffpass'==False).is_empty()

    def test_main_interms(self, main_bcorr_interm_params, main_bcorr_interm_signals):
        pass
