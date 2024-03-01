import numpy as np
import polars as pl
from hplc_py.common.definitions import X, X_IDX
from hplc_py.map_windows.definitions import W_IDX, W_TYPE

from hplc_py.map_peaks.definitions import MAXIMA, P_IDX, KEY_WIDTH_WHH

KEY_LB: str = "lb"
KEY_UB: str = "ub"
PARAM: str = "param"
KEY_P0: str = "p0"
KEY_SKEW: str = "skew"
MAXIMA: str = "maxima"
WIDTH: str = "width"
SKEW: str = "skew"
KEY_UNMIXED: str = "unmixed"
KEY_RECON: str = "recon"
KEY_POPT_IDX: str = "popt_idx"
KEY_POPT: str = "popt"
VALUE: str = "value"
KEY_WHH_WIDTH_HALF: str = KEY_WIDTH_WHH + "_half"
# bounds

VAL_SKEW_LB_SCALAR: float = -np.inf
VAL_SKEW_UB_SCALAR: float = np.inf
VAL_AMP_LB_MULT = 0.1
VAL_AMP_UP_MULT = 5

# Peak Report

KEY_AREA_UNMIXED = "area_unmixed"
KEY_MAXIMA_UNMIXED = "maxima_unmixed"
KEY_RETENTION_TIME = "retention_time"

# fit assess scores

KEY_RTOL: str = "rtol"
FTOL_KEY: str = "ftol"
VAL_RTOL: float = 0.01
VAL_FTOL: float = 0.01

KEY_TIME_START: str = "time_start"
KEY_TIME_END: str = "time_end"
KEY_AREA_MIXED: str = "area_mixed"
VAR_UNMIXED_KEY: str = "var_unmixed"
KEY_VAR_MIXED: str = "var_mixed"
KEY_MEAN_MIXED: str = "mean_mixed"
KEY_FANO_MIXED: str = "fano_mixed"
KEY_MEAN_FANO: str = "fano_mean"
KEY_SCORE_RECON: str = "score_recon"
KEY_TOLCHECK: str = "tolcheck"
KEY_TOLPASS: str = "tolpass"
VAL_W_TYPE_PEAK: str = "peak"
VAL_W_TYPE_INTERPEAK: str = "interpeak"
KEY_FANO_DIV: str = "div_fano"
KEY_FANOPASS: str = "fanopass"
KEY_STATUS: str = "status"

VAL_STATUS_VALID: str = "valid"
VAL_STATUS_NEEDS_REVIEW: str = "needs review"
VAL_STATUS_INVALID: str = "invalid"

KEY_GRADE = "grade"
KEY_COLOR = "color"

VAL_GRADE_VALID = "A+, success"
VAL_GRADE_INVALID = "F, failed"
VAL_GRADE_NEEDS_REVIEW = "C-, needs review"

# colored highlighting
VAL_COLOR_VALID = "black, on_green"
VAL_COLOR_INVALID = "black, on_red"
VAL_COLOR_NEEDS_REVIEW = "black, on_yellow"

LOC = "loc"

categories = [
    MAXIMA,
    LOC,
    WIDTH,
    SKEW,
]

p0_param_cats = pl.Enum(categories)

ACTUAL: str = "actual"