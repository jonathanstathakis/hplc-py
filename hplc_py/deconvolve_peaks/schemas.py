"""
Schemas and Fields relating to the deconvolution submodule.
"""

import pandas as pd
import pandera as pa

from ..common_schemas import BaseDF, HPLCBaseConfig, X_field_kwargs, X_idx_field_kwargs, BaseConfig, w_idx_field, p_idx_field, X_idx_field
from .typing import p0_param_cats

param_cat_field = pa.Field(isin=p0_param_cats.categories.to_list())

p0_field = pa.Field()

maxima_field = pa.Field()

skew_field = pa.Field()

# used for PSignals, a long frame containing the individual peak signals vstacked
X_idx_field_not_unique_kwargs = {k:v for k, v in X_idx_field_kwargs.items() if k not in ["unique"]} | dict(unique=False)

p_idx_field_duplicatable = p_idx_field.set_property("unique",False)
w_idx_field_duplicatable = p_idx_field.set_property("unique",False)

unmixed_field = pa.Field()


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

    w_idx: int = w_idx_field_duplicatable
    p_idx: int = p_idx_field_duplicatable
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
    w_idx: int = w_idx_field_duplicatable
    p_idx: int = p_idx_field_duplicatable
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

    w_idx: int = w_idx_field_duplicatable
    p_idx: int = p_idx_field_duplicatable
    maxima: float = maxima_field
    loc: float = pa.Field(**X_idx_field_kwargs)
    width: float = pa.Field()
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

    p_idx: int = p_idx_field_duplicatable
    X_idx: int = pa.Field(**X_idx_field_not_unique_kwargs)
    unmixed: float = unmixed_field

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PSignals"
        coerce = True

recon_field = pa.Field(**X_field_kwargs)

class RSignal(BaseDF):
    X_idx: int = X_idx_field
    recon: float = recon_field

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "RSignal"
        coerce = True
        description = (
            "The reconstituted signal, summation of the individual peak signals"
        )


class PReport(Popt):
    area_unmixed: float
    maxima_unmixed: float

    class Config(HPLCBaseConfig):
        strict = True
        ordered = False
        name = "PReport"
        coerce = True
