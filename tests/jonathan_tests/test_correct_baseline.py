from hplc_py import AMPCORR
from hplc_py.baseline_correct.correct_baseline import CorrectBaseline
from hplc_py.hplc_py_typing.hplc_py_typing import FloatArray
from hplc_py.quant import Chromatogram
from tests.jonathan_tests.conftest import AssChromResults


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
    def debug_bcorr_df_main(self, acr: AssChromResults):
        """
        ['timestep', 'shift', 'n_iter', 'signal', 's_compressed', 's_compressed_prime', 'inv_tform', 'y_corrected', 'background']
        """
        return acr.tables["bcorr_dbug_tbl"]

    @pytest.fixture
    def target_s_compressed_prime(
        self,
        debug_bcorr_df_main: Any,
    ):
        return debug_bcorr_df_main["s_compressed_prime"].to_numpy(np.float64)

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

    @pytest.fixture
    def s_compressed_main(
        self,
        acr: AssChromResults,
    ):
        s_compressed = acr.tables["bcorr_dbug_tbl"]["s_compressed"]

        return s_compressed

    def test_s_compressed_against_main(
        self,
        s_compressed: NDArray[float64],
        s_compressed_main: Any,
    ):
        """
        test calculated s_compressed against the main version
        """
        assert all(np.equal(s_compressed, s_compressed_main))

    def test_s_compressed_against_dbug_tbl(
        self,
        s_compressed: FloatArray,
        debug_bcorr_df: pd.DataFrame,
    ):
        """ """

        diff_tol = 5e-10

        s_comp_df = pd.DataFrame(
            {
                "main": debug_bcorr_df.s_compressed,
                "mine": s_compressed,
                "diff_tol": diff_tol,
            }
        )

        s_comp_df["diff"] = s_comp_df["main"] - s_comp_df["mine"]

        s_comp_df["is_diff"] = s_comp_df["diff"].abs() > s_comp_df["diff_tol"]

        if s_comp_df["is_diff"].any():
            plt.plot(s_compressed, label="s_compressed")
            plt.plot(debug_bcorr_df.s_compressed, label="debug tbl")
            plt.suptitle("Divergent S compressed series")
            plt.legend()
            plt.show()
            raise ValueError(
                f"my s_compressed is divergent from main s_compressed in {(s_comp_df['is_diff']==True).shape[0]} elements above a threshold of {diff_tol}"
            )

    def test_amp_raw_equals_main(
        self,
        amp_raw: FloatArray,
        amp_raw_main: FloatArray,
    ):
        assert np.all(np.equal(amp_raw, amp_raw_main))

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

    def test_s_compressed_prime_set(self, s_compressed_prime: FloatArray):
        pass

    def test_amp_compressed_prime_against_main(
        self,
        s_compressed_prime: FloatArray,
        target_s_compressed_prime: FloatArray,
    ):
        if not np.all(s_compressed_prime == target_s_compressed_prime):
            raise ValueError("`amp_compressed_prime` does not equal target")
        return None

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

        y = skewnorm.pdf(x) * np.power(np.max(amp_raw), 2) #type: ignore

        added_baseline = amp_raw + y

        baseline_auc = integrate.trapezoid(added_baseline, time)

        bcorr_auc = integrate.trapezoid(amp_bcorr, time)

        assert baseline_auc > bcorr_auc

    @pytest.fixture
    def debug_bcorr_df(
        self,
        bcorred_cb: CorrectBaseline,
    ):
        debug_df = bcorred_cb._debug_bcorr_df

        return debug_df

    @pytest.mark.skip
    def test_debug_bcorr_df_compare_s_compressed_prime(
        self, s_compressed_prime: NDArray[float64], debug_bcorr_df: DataFrame
    ):
        """
        I am expecting these two series to be identical, however they currently are not. the debug df is the same as the target.
        """
        diff_tol = 1e-10
        prime_df = pd.DataFrame(
            {
                "mine": s_compressed_prime,
                "main": debug_bcorr_df["s_compressed_prime"],
                "diff_tol": diff_tol,
            }
        )

        prime_df["diff"] = prime_df["main"] - prime_df["mine"]

        prime_df["is_diff"] = prime_df["diff"].abs() > prime_df["diff_tol"]

        if prime_df["is_diff"].any():
            plt.plot(debug_bcorr_df["s_compressed_prime"], label="debug series")
            plt.plot(s_compressed_prime, label="isolated")
            plt.legend()
            plt.suptitle("divergent s_comp_prime series")
            plt.show()

            raise ValueError(
                f"Divergence greater than {diff_tol} detected between my s_compressed_prime and main over {(prime_df['is_diff']==True).shape[0]} elements"
            )

    def test_compare_timestep(self, timestep: float, debug_bcorr_df: DataFrame):
        difference = timestep - debug_bcorr_df.iloc[0]

        return None