from hplc_py.common.definitions import X, X_IDX
from typing import TypedDict


KEY_SIGNAL = "signal"
KEY_BACKGROUND = "background"
KEY_CORRECTED = "corrected"
KEY_RAW = "raw"
N_ITER = 250


class BlineKwargs(TypedDict, total=False):
    n_iter: int
    window_size: int
    verbose: bool
