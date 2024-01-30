from hplc_py.hplc_py_typing.hplc_py_typing import PSignals, FitAssessScores
from hplc_py.map_signals.map_windows import WindowedSignal
from typing import Any

from pandera.typing import DataFrame
import pandera as pa
import polars as pl
import pytest

from hplc_py.fit_assessment import FitAssessment


def test_unmixed_df_exec(psignals: Any):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.relplot(psignals, x="time", y="amp_unmixed", hue="p_idx", kind="line")
    plt.show()

    pass


class TestScores:
    @pytest.fixture
    def rtol(
        self,
    ):
        return 1e-2

    @pytest.fixture
    def ftol(self):
        return 1e-2

    @pytest.fixture
    def fa(
        self,
    ) -> FitAssessment:
        fa = FitAssessment()
        return fa

    @pytest.fixture
    def scores(
        self,
        fa: FitAssessment,
        psignals: DataFrame[PSignals],
        ws: DataFrame[WindowedSignal],
        rtol: float,
        ftol: float,
    ) -> DataFrame:
        scores = fa.calc_wdw_aggs(ws, psignals, rtol, ftol)

        return scores

    @pa.check_types
    def test_scores_exec(
        self,
        scores: DataFrame[FitAssessScores],
    ):
        pass

    @pa.check_types
    def test_compare_scores(
        self,
        scores: DataFrame[FitAssessScores],
        main_scores,
    ):
        my_scores = pl.from_pandas(scores)
        main_scores = main_scores.collect().sort("window_type", "window_id")
        idx_cols = ["window_type", "window_id"]

        my_scores = (
            my_scores.rename(
                {
                    "w_type": "window_type",
                    "w_idx": "window_id",
                    "mixed_var": "signal_variance",
                    "mixed_mean": "signal_mean",
                    "mixed_fano": "signal_fano_factor",
                    "recon_score": "reconstruction_score",
                }
            )
            .select(
                pl.exclude(["tolpass", "u_peak_fano", "fano_div", "fanopass", "status"])
            )
            .sort("window_type", "window_id")
        )

        # melt
        my_scores = my_scores.melt(
            id_vars=idx_cols, variable_name="msnt", value_name="mine"
        )

        main_scores = main_scores.melt(
            id_vars=idx_cols, variable_name="msnt", value_name="main"
        )

        df = my_scores.join(
            main_scores, on=["window_type", "window_id", "msnt"], how="inner"
        )

        import polars.selectors as ps

        df = (
            df.with_columns(ps.float().round(10))
            .with_columns(
                diff=(pl.col("mine") - pl.col("main")).abs(),
                tol=pl.lit(0.05),
                hz_av=pl.sum_horizontal("mine", "main") / 2,
            )
            .with_columns(
                act_tol=pl.col("hz_av") * pl.col("tol"),
            )
            .with_columns(
                tolpass=pl.col("diff") <= pl.col("act_tol"),
            )
        )

        df = (
            df.select(pl.exclude("tolpass", "act_tol"))
            .fill_nan(0)
            .sort(
                "diff",
                descending=True,
            )
        )

    @pa.check_types
    def test_fit_report_print(
        self,
        fa: FitAssessment,
        scores: DataFrame[FitAssessScores],
    ):
        
        pass

