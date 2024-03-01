from hplc_py.deconvolution.schemas import X_Windowed_With_Recon
from hplc_py.deconvolution.schemas import FitAssessScores
from typing import Any

from pandera.typing import DataFrame
import pandera as pa
import polars as pl
import pytest
from hplc_py.map_windows.schemas import X_Windowed

from hplc_py.deconvolution.fit_assessment import calc_fit_scores

from hplc_py.deconvolution.fit_assessment import (
    FitAssessment,
    get_grading_colors_frame,
    get_grading_frame,
)

from hplc_py.deconvolution.deconvolution import PeakDeconvolver

from hplc_py.common.definitions import X
from hplc_py.map_windows.definitions import W_TYPE, W_IDX

from hplc_py.deconvolution.definitions import (
    KEY_AREA_UNMIXED,
    VAL_COLOR_INVALID,
    VAL_COLOR_NEEDS_REVIEW,
    VAL_COLOR_VALID,
    VAL_GRADE_INVALID,
    VAL_GRADE_NEEDS_REVIEW,
    VAL_GRADE_VALID,
    KEY_TIME_START,
    KEY_TIME_END,
    KEY_AREA_MIXED,
    KEY_RECON,
    KEY_VAR_MIXED,
    VAR_UNMIXED_KEY,
    KEY_MEAN_MIXED,
    KEY_MEAN_FANO,
    KEY_FANO_MIXED,
    KEY_SCORE_RECON,
    KEY_TOLCHECK,
    KEY_TOLPASS,
    KEY_TOLPASS,
    VAL_W_TYPE_PEAK,
    VAL_W_TYPE_INTERPEAK,
    KEY_FANO_DIV,
    KEY_FANOPASS,
    KEY_STATUS,
    VAL_STATUS_VALID,
    VAL_STATUS_NEEDS_REVIEW,
    VAL_STATUS_INVALID,
    VAL_RTOL,
    VAL_FTOL,
    W_IDX,
    X_IDX,
    KEY_RTOL,
    KEY_COLOR,
    KEY_GRADE,
)


@pytest.fixture
def fa() -> FitAssessment:
    fa = FitAssessment()
    return fa


@pytest.fixture
def grading_color_frame():
    df = get_grading_colors_frame(
        status_key=KEY_STATUS,
        color_key=KEY_COLOR,
        status_val_valid=VAL_STATUS_VALID,
        status_val_needs_review=VAL_STATUS_NEEDS_REVIEW,
        status_val_invalid=VAL_STATUS_INVALID,
        color_val_valid=VAL_COLOR_VALID,
        color_val_needs_review=VAL_COLOR_NEEDS_REVIEW,
        color_val_invalid=VAL_COLOR_INVALID,
    )

    return df


@pytest.fixture
def grading_frame():
    df = get_grading_frame(
        status_key=KEY_STATUS,
        grade_key=KEY_GRADE,
        status_val_valid=VAL_STATUS_VALID,
        status_val_invalid=VAL_STATUS_INVALID,
        status_val_needs_review=VAL_STATUS_NEEDS_REVIEW,
        grade_val_valid=VAL_GRADE_VALID,
        grade_val_invalid=VAL_GRADE_INVALID,
        grade_val_needs_review=VAL_GRADE_NEEDS_REVIEW,
    )
    return df


@pytest.fixture
def scores(
    pdc_tform: PeakDeconvolver,
    grading_color_frame: pl.DataFrame,
    grading_frame: pl.DataFrame,
) -> DataFrame:

    X_w_with_recon: DataFrame[X_Windowed_With_Recon] = pdc_tform.X_w_with_recon

    scores = calc_fit_scores(
        X_w_with_recon=X_w_with_recon,
        rtol=VAL_RTOL,
        ftol=VAL_FTOL,
        w_type_key=W_TYPE,
        key_w_idx=W_IDX,
        key_X_idx=X_IDX,
        key_X=X,
        key_recon=KEY_RECON,
        key_time_start=KEY_TIME_START,
        key_time_end=KEY_TIME_END,
        key_area_mixed=KEY_AREA_MIXED,
        key_area_unmixed=KEY_AREA_UNMIXED,
        key_var_mixed=KEY_VAR_MIXED,
        key_mean_mixed=KEY_MEAN_MIXED,
        key_fano_mixed=KEY_FANO_MIXED,
        key_score_recon=KEY_SCORE_RECON,
        key_rtol=KEY_RTOL,
        key_tolcheck=KEY_TOLCHECK,
        key_tolpass=KEY_TOLPASS,
        key_mean_fano=KEY_MEAN_FANO,
        val_w_type_peak=VAL_W_TYPE_PEAK,
        key_fano_div=KEY_FANO_DIV,
        key_fanopass=KEY_FANOPASS,
        key_status=KEY_STATUS,
        grading_frame=grading_frame,
        grading_color_frame=grading_color_frame,
        w_type_peak_val=VAL_W_TYPE_PEAK,
        val_w_type_interpeak=VAL_W_TYPE_INTERPEAK,
        val_status_valid=VAL_STATUS_VALID,
        val_status_needs_review=VAL_STATUS_NEEDS_REVIEW,
        val_status_invalid=VAL_STATUS_INVALID,
    )
    return scores


@pa.check_types
def test_scores_exec(
    scores: DataFrame[FitAssessScores],
):
    pass


@pa.check_types
def test_fit_report_print(
    fa: FitAssessment,
    scores: DataFrame[FitAssessScores],
):

    pass
