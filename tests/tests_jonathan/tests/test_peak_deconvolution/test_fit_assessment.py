from hplc_py.deconvolution.schemas import X_Windowed_With_Recon
from hplc_py.deconvolution.schemas import FitAssessScores
from typing import Any

from pandera.typing import DataFrame
import pandera as pa
import polars as pl
import pytest
from hplc_py.map_windows.schemas import X_Windowed

from hplc_py.deconvolution.fit_assessment import calc_fit_report

from hplc_py.deconvolution.fit_assessment import (
    FitAssessment,
    get_grading_colors_frame,
    get_grading_frame,
)

from hplc_py.deconvolution.deconvolution import PeakDeconvolver

from hplc_py.common.definitions import X
from hplc_py.map_windows.definitions import W_TYPE, W_IDX

from hplc_py.deconvolution.definitions import (
    area_unmixed,
    VAL_COLOR_INVALID,
    VAL_COLOR_NEEDS_REVIEW,
    VAL_COLOR_VALID,
    VAL_GRADE_INVALID,
    VAL_GRADE_NEEDS_REVIEW,
    VAL_GRADE_VALID,
    time_start,
    time_end,
    area_mixed,
    KEY_RECON,
    var_mixed,
    VAR_UNMIXED_KEY,
    mean_mixed,
    mean_fano,
    fano_mixed,
    score_recon,
    tolcheck,
    tolpass,
    tolpass,
    w_type_peak,
    w_type_interpeak,
    fano_div,
    fano_pass,
    status,
    val_status_valid,
    val_status_needs_review,
    val_status_invalid,
    VAL_RTOL,
    VAL_FTOL,
    W_IDX,
    IDX,
    rtol,
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
        status_key=status,
        color_key=KEY_COLOR,
        status_val_valid=val_status_valid,
        status_val_needs_review=val_status_needs_review,
        status_val_invalid=val_status_invalid,
        color_val_valid=VAL_COLOR_VALID,
        color_val_needs_review=VAL_COLOR_NEEDS_REVIEW,
        color_val_invalid=VAL_COLOR_INVALID,
    )

    return df


@pytest.fixture
def grading_frame():
    df = get_grading_frame(
        status_key=status,
        grade_key=KEY_GRADE,
        status_val_valid=val_status_valid,
        status_val_invalid=val_status_invalid,
        status_val_needs_review=val_status_needs_review,
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

    scores = calc_fit_report(
        data_=X_w_with_recon,
        rtol=VAL_RTOL,
        ftol=VAL_FTOL,
        w_type_key=W_TYPE,
        key_w_idx=W_IDX,
        key_X_idx=IDX,
        key_X=X,
        key_recon=KEY_RECON,
        key_time_start=time_start,
        key_time_end=time_end,
        key_area_mixed=area_mixed,
        key_area_unmixed=area_unmixed,
        key_var_mixed=var_mixed,
        key_mean_mixed=mean_mixed,
        key_fano_mixed=fano_mixed,
        key_score_recon=score_recon,
        key_rtol=rtol,
        key_tolcheck=tolcheck,
        key_tolpass=tolpass,
        key_mean_fano=mean_fano,
        val_w_type_peak=w_type_peak,
        key_fano_div=fano_div,
        key_fanopass=fano_pass,
        key_status=status,
        grading_frame=grading_frame,
        grading_color_frame=grading_color_frame,
        w_type_peak_val=w_type_peak,
        val_w_type_interpeak=w_type_interpeak,
        val_status_valid=val_status_valid,
        val_status_needs_review=val_status_needs_review,
        val_status_invalid=val_status_invalid,
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
