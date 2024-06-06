from hplc_py.common.definitions import X, IDX
from typing import TypedDict


KEY_SIGNAL = "signal"
KEY_BACKGROUND = "background"
KEY_CORRECTED = "corrected"
KEY_RAW = "raw"
N_ITER = 250


class BlineKwargs(TypedDict, total=False):
    n_iter: int
    verbose: bool
