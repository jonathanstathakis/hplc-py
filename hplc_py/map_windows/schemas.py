"""
Pandera typing for `map_windows`. Fields are parametrized as much as possible and reused, rather than using inheritance, which is a tad brittle in the current version of Pandera - the main problem is that column ordering is inflexible and dependant on MRO.
"""

import pandera as pa
from pandera.dtypes import String

from ..common_schemas import BaseDF
from ..common_schemas import HPLCBaseConfig
import pandas as pd

from ..common_schemas import (
    p_idx_field,
    w_idx_field,
    left_base_field,
    right_base_field,
    w_type_field,
    X_idx_field,
    X_field,
    w_idx_field_nullable
)

class WindowPeakMap(pa.DataFrameModel):
    p_idx: int = p_idx_field
    w_idx: int = w_idx_field

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowPeakMap"
        coerce = True


class WindowedPeakIntervals(pa.DataFrameModel):
    w_idx: int = w_idx_field
    p_idx: int = p_idx_field
    left: int = left_base_field
    right: int = right_base_field

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowedPeakIntervals"
        coerce = True

class WindowBounds(pa.DataFrameModel):
    w_idx: int = w_idx_field
    left: int = left_base_field
    right: int = right_base_field

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowBounds"
        coerce = True


class InterpeakWindowStarts(pa.DataFrameModel):
    w_type: String = w_type_field
    w_idx: int = w_idx_field
    X_idx: int = X_idx_field
    X: float = X_field
    
    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "InterpeakWindowStarts"
        coerce = True


class PeakIntervalBounds(pa.DataFrameModel):
    p_idx: int = p_idx_field
    left: int = left_base_field
    right: int = right_base_field

class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PeakIntervalBounds"
        coerce = True


class WindowedPeakIntervalBounds(pa.DataFrameModel):
    w_idx: int = w_idx_field
    p_idx: int = p_idx_field
    left: int = left_base_field
    right: int = right_base_field

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowedPeakIntervalBounds"
        coerce = True


class PeakWindows(BaseDF):
    w_type: pd.StringDtype
    w_idx: pd.Int64Dtype
    X_idx: pd.Int64Dtype

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PeakWindows"
        coerce = True


class X_PeakWindowed(BaseDF):
    """
    peak windowed time dataframe, with placeholders for nonpeak regions. An intermediate frame prior to full mapping
    """

    w_type: String = w_type_field
    w_idx: pd.Int64Dtype = w_idx_field_nullable
    X_idx: int = X_idx_field
    X: float

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "X_PeakWindowed"
        coerce = True


class X_Windowed(X_PeakWindowed):
    w_idx: int = pa.Field(ge=-9999, le=10)

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "X_Windowed"
        coerce = True
