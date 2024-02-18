
from .definitions import MAXIMA_KEY, X_IDX_KEY, WHH_WIDTH_HALF_KEY, SKEW_KEY

import polars as pl

categories = [
    MAXIMA_KEY,
    X_IDX_KEY,
    WHH_WIDTH_HALF_KEY,
    SKEW_KEY,
]

p0_param_cats = pl.Enum(categories)
