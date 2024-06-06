from dataclasses import dataclass
import numpy as np
import polars as pl
from hplc_py.common.definitions import X, IDX
from hplc_py.map_windows.definitions import W_IDX, W_TYPE

from hplc_py.map_peaks.definitions import MAXIMA, P_IDX, KEY_WIDTH_WHH

KEY_LB: str = "lb"
KEY_UB: str = "ub"
PARAM: str = "param"
KEY_P0: str = "p0"
KEY_SKEW: str = "skew"
MAXIMA: str = "maxima"
SCALE: str = "scale"
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
VAL_AMP_UP_MULT = 10

# Peak Report

area_unmixed = "area_unmixed"
KEY_MAXIMA_UNMIXED = "maxima_unmixed"
KEY_RETENTION_TIME = "retention_time"

# fit assess scores

rtol: str = "rtol"
FTOL_KEY: str = "ftol"
VAL_RTOL: float = 0.01
VAL_FTOL: float = 0.01

time_start: str = "time_start"
time_end: str = "time_end"
area_mixed: str = "area_mixed"
VAR_UNMIXED_KEY: str = "var_unmixed"
var_mixed: str = "var_mixed"
mean_mixed: str = "mean_mixed"
fano_mixed: str = "fano_mixed"
mean_fano: str = "fano_mean"
score_recon: str = "score_recon"
tolcheck: str = "tolcheck"
tolpass: str = "tolpass"
w_type_peak: str = "peak"
w_type_interpeak: str = "interpeak"
fano_div: str = "div_fano"
fano_pass: str = "fanopass"
status: str = "status"

val_status_valid: str = "valid"
val_status_needs_review: str = "needs review"
val_status_invalid: str = "invalid"

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
    SCALE,
    SKEW,
]

p0_param_cats = pl.Enum(categories)

ACTUAL: str = "actual"

MSNT: str = "msnt"


@dataclass(frozen=True)
class KeysTblMixedSignal:
    W_TYPE: str
    W_IDX: str
    SIGNAL: str
    AMPLITUDE: str
    RECON: str
    MIXED: str


SIGNAL = "signal"
AMPLITUDE = "amplitude"
RECON = "recon"
MIXED = "mixed"

keys_tbl_mixed_signal = KeysTblMixedSignal(
    W_TYPE=W_TYPE,
    W_IDX=W_IDX,
    SIGNAL=SIGNAL,
    AMPLITUDE=AMPLITUDE,
    RECON=RECON,
    MIXED=MIXED,
)
