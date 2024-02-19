from typing import Any
import pandas as pd

from ..common_schemas import BaseDF, HPLCBaseConfig
from ..common_schemas import (
    p_idx_field,
    X_idx_field,
    X_field_kwargs,
    X_idx_field_kwargs,
)

import pandera as pa

prom_field = pa.Field()

whh_width_field = pa.Field()
whh_rel_height_field = pa.Field()

pb_rel_height_field = pa.Field()
pb_width_field = pa.Field()

# X_idx values but can be duplicated
prom_field_kwargs: dict[str, Any] = {
    k: v for k, v in X_idx_field_kwargs.items() if k !="unique"
} | dict(unique=False)


class FindPeaks(BaseDF):
    p_idx: int = p_idx_field
    X_idx: int = X_idx_field
    maxima: float = pa.Field(**X_field_kwargs)
    prom: float = prom_field
    prom_left: int = pa.Field(**prom_field_kwargs)
    prom_right: int = pa.Field(**prom_field_kwargs)

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "FindPeaks"
        coerce = True

# bounds of peak widths have same properties as X_idx but can be repeated
widths_intvl_bounds_kwargs_repeatable = {k:v for k, v in X_idx_field_kwargs.items() if k not in ["unique"]} | dict(unique=False)

class WHH(BaseDF):
    p_idx: int = p_idx_field
    whh_rel_height: pd.Float64Dtype = whh_rel_height_field
    whh_width: float = whh_width_field
    whh_height: float = pa.Field(**X_field_kwargs)
    whh_left: float = pa.Field(**widths_intvl_bounds_kwargs_repeatable)
    whh_right: float = pa.Field(**widths_intvl_bounds_kwargs_repeatable)

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WHH"
        coerce = True


class PeakBases(BaseDF):
    p_idx: int = p_idx_field
    pb_rel_height: float = pb_rel_height_field
    pb_width: float = pb_width_field
    pb_height: float = pa.Field(**X_field_kwargs)
    pb_left: float = pa.Field(**widths_intvl_bounds_kwargs_repeatable)
    pb_right: float = pa.Field(**widths_intvl_bounds_kwargs_repeatable)

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PeakBases"
        coerce = True


class PeakMapWideColored(
    PeakBases,
    WHH,
    FindPeaks,
):
    color: object

    class Config(HPLCBaseConfig):
        col_a_less_than_col_b = {"col_a": "whh_width", "col_b": "pb_width"}
        name = "PeakMapWide"


class PeakMapWide(
    PeakBases,
    WHH,
    FindPeaks,
):
    class Config(HPLCBaseConfig):
        col_a_less_than_col_b = {"col_a": "whh_width", "col_b": "pb_width"}
        name = "PeakMapWide"
