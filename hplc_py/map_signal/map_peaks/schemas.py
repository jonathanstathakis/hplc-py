from typing import Any
from pandera import polars as papl

from hplc_py.hplc_py_typing.custom_checks import col_a_less_than_col_b  # noqa: F401
from hplc_py.common.common_schemas import BaseDF, HPLCBaseConfig
from hplc_py.common.common_schemas import (
    p_idx_field,
    X_field_kwargs,
    idx_field_kwargs,
)

import pandera as pa

from . import definitions as mp_defs

prom_field = pa.Field()

whh_width_field = pa.Field()
whh_rel_height_field = pa.Field()

pb_rel_height_field = pa.Field()
pb_width_field = pa.Field()

# X_idx values but can be duplicated
prom_field_kwargs: dict[str, Any] = {
    k: v for k, v in idx_field_kwargs.items() if k != "unique"
} | dict(unique=False)


class FindPeaks(papl.DataFrameModel):
    p_idx: int = p_idx_field
    idx: int
    maxima: float = pa.Field(**X_field_kwargs)
    prominence: float = prom_field
    left_prom: int = pa.Field(**prom_field_kwargs)
    right_prom: int = pa.Field(**prom_field_kwargs)

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "FindPeaks"
        coerce = True


# bounds of peak widths have same properties as X_idx but can be repeated
widths_intvl_bounds_kwargs_repeatable = {
    k: v for k, v in idx_field_kwargs.items() if k not in ["unique"]
} | dict(unique=False)


class WHH(papl.DataFrameModel):
    p_idx: int = p_idx_field
    width_whh: float = whh_width_field
    height_whh: float = pa.Field(**X_field_kwargs)
    left_whh: float = pa.Field(**widths_intvl_bounds_kwargs_repeatable)
    right_whh: float = pa.Field(**widths_intvl_bounds_kwargs_repeatable)

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WHH"
        coerce = True


class PeakBases(papl.DataFrameModel):
    p_idx: int = p_idx_field
    width_pb: float = pb_width_field
    height_pb: float = pa.Field(**X_field_kwargs)
    left_pb: float = pa.Field(**widths_intvl_bounds_kwargs_repeatable)
    right_pb: float = pa.Field(**widths_intvl_bounds_kwargs_repeatable)

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
        col_a_less_than_col_b = {
            "col_a": mp_defs.KEY_WIDTH_WHH,
            "col_b": mp_defs.KEY_WIDTH_PB,
        }
        name = "PeakMapWide"


class PeakMap(
    PeakBases,
    WHH,
    FindPeaks,
):
    class Config(HPLCBaseConfig):
        col_a_less_than_col_b = {
            "col_a": mp_defs.KEY_WIDTH_WHH,
            "col_b": mp_defs.KEY_WIDTH_PB,
        }
        name = "PeakMapWide"


# going from wide peak map to long peak map requires that p_idx is now repeatable

p_idx_field_repeatable_kwargs = {
    k: v for k, v in idx_field_kwargs.items() if k not in ["unique"]
} | dict(unique=False)

p_idx_field_repeatable = pa.Field(**p_idx_field_repeatable_kwargs)


class Maxima(papl.DataFrameModel):
    p_idx: int = p_idx_field_repeatable
    loc: str = pa.Field(isin=[mp_defs.MAXIMA])
    dim: str = pa.Field(isin=[mp_defs.X, mp_defs.IDX])
    value: float = pa.Field()

    class Config:
        strict = True
        ordered = True
        description = "A normalized table containing the peak maxima as detected by `scipy.signal.find_peaks`"


class ContourLineBoundsSchema(papl.DataFrameModel):
    p_idx: int = p_idx_field_repeatable
    loc: str = pa.Field(isin=[mp_defs.KEY_LEFT, mp_defs.KEY_RIGHT])
    msnt: str = pa.Field(isin=[mp_defs.KEY_PB, mp_defs.KEY_PROM, mp_defs.KEY_WHH])
    dim: str = pa.Field(isin=[mp_defs.X, mp_defs.KEY_IDX_ROUNDED])
    value: float = pa.Field()

    class Config:
        strict = True
        ordered = True
        description = "A normalized table containing the left and right bounds of the contour lines used to measure the peak prominence, and widths at 0.5 and 1 rel height. A collection of the output of three seperate calculations."


class Widths(papl.DataFrameModel):
    p_idx: int = p_idx_field_repeatable
    msnt: str = pa.Field(isin=[mp_defs.KEY_WIDTH_WHH, mp_defs.KEY_WIDTH_PB])
    value: float = pa.Field(ge=0)

    class Config:
        strict = True
        ordered = True
        description = "A normalised table containing the widths measurements, i.e. for WHH and peakre bases"
