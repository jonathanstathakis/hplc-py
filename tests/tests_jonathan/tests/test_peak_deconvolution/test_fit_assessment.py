# from hplc_py.hplc_py_typing.hplc_py_typing import FitAssessScores
# from typing import Any

# from pandera.typing import DataFrame
# import pandera as pa
# import polars as pl
# import pytest
# from hplc_py.map_windows.schemas import X_Windowed

# from hplc_py.deconvolve_peaks.fit_assessment import calc_wdw_aggs

# from hplc_py.deconvolve_peaks.fit_assessment import FitAssessment


# @pytest.fixture
# def fa() -> FitAssessment:
#     fa = FitAssessment()
#     return fa


# @pytest.fixture
# def scores(
#     asschrom_ws: DataFrame[X_Windowed],
#     rtol: float,
#     ftol: float,
# ) -> DataFrame:
#     scores = calc_wdw_aggs(asschrom_ws, rtol, ftol)

#     return scores


# def test_unmixed_df_exec(psignals: Any):
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     sns.relplot(psignals, x="time", y="amp_unmixed", hue="p_idx", kind="line")
#     plt.show()

#     pass


# class TestScores:
#     @pytest.fixture
#     def rtol(
#         self,
#     ):
#         return 1e-2

#     @pytest.fixture
#     def ftol(self):
#         return 1e-2

#     @pa.check_types
#     def test_scores_exec(
#         self,
#         scores: DataFrame[FitAssessScores],
#     ):
#         pass

#     @pa.check_types
#     def test_fit_report_print(
#         self,
#         fa: FitAssessment,
#         scores: DataFrame[FitAssessScores],
#     ):

#         pass
