"""
Pandera typing for `map_windows`. Fields are parametrized as much as possible and reused, rather than using inheritance, which is a tad brittle in the current version of Pandera - the main problem is that column ordering is inflexible and dependant on MRO.
"""

import pandera as pa
from pandera.dtypes import String
from ..hplc_py_typing.hplc_py_typing import BaseDF, HPLCBaseConfig
import pandas as pd

INTERPEAK_LABEL, PEAK_LABEL = "interpeak", "peak"

w_type_values = [INTERPEAK_LABEL, PEAK_LABEL]

# these are defined as closed intervals, i.e. ge, le
p_idx_min = 0
p_idx_max = 100
X_field_min = -100
X_field_max = 5000
X_idx_field_min = 0
X_idx_field_max = 5000

X_idx_min = 0
X_idx_max = 5000

left_base_field_min = X_idx_min
left_base_field_max = X_idx_max
right_base_field_min = X_idx_min
right_base_field_max = X_idx_max

w_idx_field_min = 0
w_idx_field_max = 100

p_idx_field = pa.Field(ge=p_idx_min, le=p_idx_max, unique=True)
X_field = pa.Field(ge=X_field_min, le=X_field_max)
X_idx_field = pa.Field(ge=X_idx_field_min, le=X_idx_field_max, unique=True)
left_base_field = pa.Field(ge=left_base_field_min, le=left_base_field_max)
right_base_field = pa.Field(ge=right_base_field_min, le=right_base_field_max)
w_idx_field = pa.Field(ge=w_idx_field_min, le=w_idx_field_max)
w_idx_field_nullable = w_idx_field.set_property('nullable', True)
w_type_field = pa.Field(isin=w_type_values)


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
