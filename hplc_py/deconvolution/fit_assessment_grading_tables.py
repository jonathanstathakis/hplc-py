import polars as pl
import hplc_py.deconvolution.definitions as defs


def get_grading_frame(
    status_key: str = defs.KEY_STATUS,
    grade_key: str = defs.KEY_GRADE,
    status_val_valid: str = defs.VAL_STATUS_VALID,
    status_val_invalid: str = defs.VAL_STATUS_INVALID,
    status_val_needs_review: str = defs.VAL_STATUS_NEEDS_REVIEW,
    grade_val_valid: str = defs.VAL_GRADE_VALID,
    grade_val_invalid: str = defs.VAL_GRADE_INVALID,
    grade_val_needs_review: str = defs.VAL_GRADE_NEEDS_REVIEW,
) -> pl.DataFrame:
    grading = pl.DataFrame(
        {
            status_key: [status_val_valid, status_val_invalid, status_val_needs_review],
            grade_key: [grade_val_valid, grade_val_invalid, grade_val_needs_review],
        }
    )
    return grading


def get_grading_colors_frame(
    status_key: str = defs.KEY_STATUS,
    color_key: str = defs.KEY_COLOR,
    status_val_valid: str = defs.VAL_STATUS_VALID,
    status_val_invalid: str = defs.VAL_STATUS_INVALID,
    status_val_needs_review: str = defs.VAL_STATUS_NEEDS_REVIEW,
    color_val_valid: str = defs.VAL_COLOR_VALID,
    color_val_invalid: str = defs.VAL_COLOR_INVALID,
    color_val_needs_review: str = defs.VAL_COLOR_NEEDS_REVIEW,
) -> pl.DataFrame:
    grading_colors = pl.DataFrame(
        {
            status_key: [status_val_valid, status_val_invalid, status_val_needs_review],
            color_key: [
                color_val_valid,
                color_val_invalid,
                color_val_needs_review,
            ],
        }
    )

    return grading_colors
