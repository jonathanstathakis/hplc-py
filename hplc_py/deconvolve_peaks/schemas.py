"""
Schemas and Fields relating to the deconvolution submodule.
"""
import pandas as pd
import pandera as pa

from ..common_schemas import BaseDF, HPLCBaseConfig
from .definitions import MAXIMA_KEY, X_IDX_KEY, WHH_WIDTH_HALF_KEY, SKEW_KEY
from ..common_schemas import BaseConfig, BaseDF, w_idx_field, p_idx_field, X_idx_field

param_cat_field = pa.Field(isin=[MAXIMA_KEY, X_IDX_KEY, WHH_WIDTH_HALF_KEY, SKEW_KEY])

p0_field = pa.Field()

maxima_field = pa.Field()

whh_width_half_field = pa.Field()

skew_field = pa.Field()

class InP0(pa.DataFrameModel):
    w_idx: int = w_idx_field
    p_idx: int = p_idx_field
    maxima: float = maxima_field
    X_idx: int = X_idx_field
    whh_width: float

    class Config(BaseConfig):
        name = "in_p0"
        ordered = False
        strict = True

class P0(BaseDF):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    w_idx: int = w_idx_field
    p_idx: int = p_idx_field
    param: pd.CategoricalDtype = param_cat_field
    p0: float = p0_field

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "P0"
        coerce = True

lb_field = pa.Field()
ub_field = pa.Field()

class Bounds(BaseDF):
    w_idx: int = w_idx_field
    p_idx: int= p_idx_field
    param: pd.CategoricalDtype = param_cat_field
    lb: float = lb_field
    ub: float = ub_field

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "Bounds"
        coerce = True


class Params(Bounds, P0):
    pass

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "Params"
        coerce = True

        col_in_lb = {"col": "p0", "col_lb": "lb"}
        col_in_ub = {"col": "p0", "col_ub": "ub"}

class Popt(BaseDF):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    w_idx: int = w_idx_field
    p_idx: int = p_idx_field
    maxima: float = maxima_field
    X_idx: float = X_idx_field
    whh_half: float = whh_width_half_field
    skew: float = skew_field

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "Popt"
        coerce = True


class PSignals(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    p_idx: int = pa.Field()
    t_idx: int = pa.Field()
    time: float = pa.Field()
    amp_unmixed: float = pa.Field()

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PSignals"
        coerce = True


class RSignal(BaseDF):
    t_idx: int
    time: float
    amp_unmixed: float

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "RSignal"
        coerce = True
        description = (
            "The reconstituted signal, summation of the individual peak signals"
        )


class PReport(Popt):
    retention_time: float
    area_unmixed: float
    maxima_unmixed: float

    class Config(HPLCBaseConfig):
        strict = True
        ordered = False
        name = "PReport"
        coerce = True
