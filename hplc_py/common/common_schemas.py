from typing import Any, Optional
import pandera as pa
from pandera.api.pandas.model_config import BaseConfig


# these are defined as closed intervals, i.e. ge, le
p_idx_min = 0
p_idx_max = 100
X_field_min = -100
X_field_max = 200
X_idx_field_min = 0
X_idx_field_max = 15000

left_base_field_min = X_idx_field_min
left_base_field_max = X_idx_field_max
right_base_field_min = X_idx_field_min
right_base_field_max = X_idx_field_max


# note: current version of pandera instantiates the field objects are persistant classes in the pandera scope, making it impossible to reuse

p_idx_field_kwargs: dict[str, Any] = dict(ge=p_idx_min, le=p_idx_max, unique=True)
p_idx_field = pa.Field(**p_idx_field_kwargs)

X_field_kwargs: dict[str, Any] = dict(ge=X_field_min, le=X_field_max)
X_field = pa.Field(**X_field_kwargs)

X_idx_field_kwargs: dict[str, Any] = dict(
    ge=X_idx_field_min, le=X_idx_field_max, unique=True
)
X_idx_field = pa.Field(**X_idx_field_kwargs)

left_base_field = pa.Field(ge=left_base_field_min, le=left_base_field_max)

right_base_field = pa.Field(ge=right_base_field_min, le=right_base_field_max)

class HPLCBaseConfig(BaseConfig):
    strict = True
    ordered = True
    name = "!!PLEASE PROVIDE NAME!!"
    coerce = True

class BaseDF(pa.DataFrameModel):
    """
    Lowest level class for basic DataFrame assumptions - for example, they will all
    contain a index named 'idx' which is the default RangedIndex
    """

    # idx: Index[int] = pa.Field(check_name=True)

    # @pa.check(
    #     "idx", name="idx_check", error="expected range index bounded by 0 and len(df)"
    # )
    # def check_is_range_index(cls, idx: Series[int]) -> bool:
    #     left_idx = pd.RangeIndex(0, len(idx) - 1)
    #     right_idx = pd.RangeIndex(idx.iloc[0], idx.iloc[-1])
    #     check = left_idx.equals(right_idx)
    #     return check

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "BaseDF"
        coerce = True


class X_Schema(pa.DataFrameModel):
    X_idx: int
    X: float
    time: Optional[float]
    X_corrected: Optional[float]

    class Config:
        strict = True
        description = "A simplistic container for the signal array"
        unique = ["X_idx"]


class RawData(BaseDF):
    """
    The base signal, with time and amplitude directions
    """

    t_idx: int
    time: float
    amp: float

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "SignalDFLoaded"
        coerce = True
