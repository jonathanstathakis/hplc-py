from numpy import float64
import pandera as pa
from pandera.api.pandas.model_config import BaseConfig
from .common_definitions import w_type_values

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

w_idx_field_nullable = w_idx_field.set_property("nullable", True)

w_type_field = pa.Field(isin=w_type_values)


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
    X: float64

    class Config:
        strict = True
        description = "A simplistic container for the signal array"
        unique = ["X_idx"]
