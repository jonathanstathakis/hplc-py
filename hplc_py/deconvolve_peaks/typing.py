
from .definitions import PARAM_VAL_LOC, PARAM_VAL_MAX, PARAM_VAL_SKEW, PARAM_VAL_WIDTH

import polars as pl

categories = [
    PARAM_VAL_MAX,
    PARAM_VAL_LOC,
    PARAM_VAL_WIDTH,
    PARAM_VAL_SKEW,
]

p0_param_cats = pl.Enum(categories)
