"""
Pandera typing for `map_windows`. Fields are parametrized as much as possible and reused, rather than using inheritance, which is a tad brittle in the current version of Pandera - the main problem is that column ordering is inflexible and dependant on MRO.
"""

import pandera.polars as pa
from pandera.dtypes import String
import pandas as pd

import hplc_py.common.common_schemas as com_schs
from hplc_py.common.definitions import LABEL_INTERPEAK, LABEL_PEAK


w_idx_field_min = 0
w_idx_field_max = 100


w_idx_field_kwargs = dict(ge=w_idx_field_min, le=w_idx_field_max)


w_idx_field = pa.Field(**w_idx_field_kwargs)


class WindowPeakMap(pa.DataFrameModel):
    p_idx: int = com_schs.p_idx_field
    w_idx: int = w_idx_field

    class Config(com_schs.HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowPeakMap"
        coerce = True


class WindowedPeakIntervals(pa.DataFrameModel):
    w_idx: int = w_idx_field
    p_idx: int = com_schs.p_idx_field
    left: int = com_schs.left_base_field
    right: int = com_schs.right_base_field

    class Config(com_schs.HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowedPeakIntervals"
        coerce = True


class WindowBounds(pa.DataFrameModel):
    w_type: str
    w_idx: int = w_idx_field
    left: int = com_schs.left_base_field
    right: int = com_schs.right_base_field

    class Config(com_schs.HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowBounds"
        coerce = True


VALUES_W_TYPE = [LABEL_INTERPEAK, LABEL_PEAK]


w_type_field = pa.Field(isin=VALUES_W_TYPE)


class InterpeakWindowStarts(pa.DataFrameModel):
    w_type: String = w_type_field
    w_idx: int = w_idx_field
    X_idx: int = com_schs.idx_field
    X: float = com_schs.X_field

    class Config(com_schs.HPLCBaseConfig):
        strict = True
        ordered = True
        name = "InterpeakWindowStarts"
        coerce = True


class PeakIntervalBounds(pa.DataFrameModel):
    p_idx: int = com_schs.p_idx_field
    left: int = com_schs.left_base_field
    right: int = com_schs.right_base_field


class Config(com_schs.HPLCBaseConfig):
    strict = True
    ordered = True
    name = "PeakIntervalBounds"
    coerce = True


class WindowedPeakIntervalBounds(pa.DataFrameModel):
    w_idx: int = w_idx_field
    p_idx: int = com_schs.p_idx_field
    left: int = com_schs.left_base_field
    right: int = com_schs.right_base_field

    class Config(com_schs.HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowedPeakIntervalBounds"
        coerce = True


from pandera.dtypes import String, Int

class PeakWindows(pa.DataFrameModel):
    w_type: String
    w_idx: int
    X_idx: int

    class Config(com_schs.HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PeakWindows"
        coerce = True


w_idx_field_kwargs_nullable = {
    k: v for k, v in w_idx_field_kwargs.items() if k != "nullable"
} | {"nullable": True}

w_idx_field_nullable = pa.Field(**w_idx_field_kwargs_nullable)


class X_PeakWindowed(pa.DataFrameModel):
    """
    peak windowed time dataframe, with placeholders for nonpeak regions. An intermediate frame prior to full mapping
    """

    w_type: String = w_type_field
    w_idx: int = w_idx_field_nullable
    idx: int = com_schs.idx_field
    X: float

    class Config(com_schs.HPLCBaseConfig):
        strict = True
        ordered = True
        name = "X_PeakWindowed"
        coerce = True


class WindowFields(pa.DataFrameModel):
    w_type: str = w_type_field
    w_idx: int = w_idx_field


class X_Windowed(X_PeakWindowed, WindowFields):
    pass

    class Config(com_schs.HPLCBaseConfig):
        strict = True
        ordered = True
        name = "X_Windowed"
        coerce = True


# New Schemas, above are stale or invalid as of 2024-03-07 14:42:21
# FIXME clear out old invalid schemas
