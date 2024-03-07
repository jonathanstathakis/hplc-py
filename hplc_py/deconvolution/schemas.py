"""
Schemas and Fields relating to the deconvolution submodule.

TODO: reduce duplication of fields by defining common abstracted classes, for example a class that has w_type, w_idx and nothing else.
TODO: change explicit imports of constants in header to import of module then access at location i.e. from `LOC` to `__defs.LOC`
"""

import pandas as pd
import pandera as pa

from typing import Any
from hplc_py.deconvolution.definitions import (
    VAL_COLOR_INVALID,
    VAL_COLOR_NEEDS_REVIEW,
    VAL_COLOR_VALID,
    VAL_GRADE_INVALID,
    VAL_GRADE_NEEDS_REVIEW,
    VAL_GRADE_VALID,
    val_status_invalid,
    val_status_needs_review,
    val_status_valid,
)
from hplc_py.map_peaks import definitions as mp_defs, schemas as mp_schs
from hplc_py.map_windows.schemas import w_idx_field

from hplc_py.map_windows.schemas import w_type_field

from hplc_py.common.common_schemas import (
    BaseDF,
    HPLCBaseConfig,
    X_field_kwargs,
    X_idx_field_kwargs,
    BaseConfig,
    p_idx_field,
    X_idx_field,
    p_idx_field_kwargs,
)
from hplc_py.map_windows.schemas import w_idx_field_kwargs
from hplc_py.deconvolution.definitions import p0_param_cats

from hplc_py.map_windows.schemas import X_Windowed
from hplc_py.map_windows import schemas as mw_schs

param_cat_field = pa.Field(isin=p0_param_cats.categories.to_list())

p0_field = pa.Field()

maxima_field = pa.Field()

skew_field = pa.Field()

# used for PSignals, a long frame containing the individual peak signals vstacked
X_idx_field_not_unique_kwargs = {
    k: v for k, v in X_idx_field_kwargs.items() if k not in ["unique"]
} | dict(unique=False)

p_idx_field_duplicatable_kwargs = {
    k: v for k, v in p_idx_field_kwargs.items() if k != "unique"
} | {"unique": False}

w_idx_field_duplicatable_kwargs = {
    k: v for k, v in w_idx_field_kwargs.items() if k != "unique"
} | {"unique": False}

p_idx_field_duplicatable = pa.Field(**p_idx_field_duplicatable_kwargs)
w_idx_field_duplicatable = pa.Field(**w_idx_field_duplicatable_kwargs)

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

    w_type: str
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
    scale: float = pa.Field()
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


class X_Windowed_With_Recon(X_Windowed):

    recon: float = pa.Field(**X_field_kwargs)

    class Config:
        name = "X_Windowed_With_Recon"
        strict = True
        ordered = True
        coerce = True


area_kwargs: dict[str, Any] = dict(ge=0)
var_kwargs: dict[str, Any] = dict()
fano_kwargs: dict[str, Any] = dict()
score_kwargs: dict[str, Any] = dict()
rtol_kwargs: dict[str, Any] = dict()
tolcheck_kwargs: dict[str, Any] = dict()
tolpass_kwargs: dict[str, Any] = dict()
fano_mean_peaks_kwargs: dict[str, Any] = dict() | fano_kwargs
div_fano_kwargs: dict[str, Any] = dict() | fano_kwargs
fanopass_kwargs: dict[str, Any] = dict()
status_kwargs: dict[str, Any] = dict(
    isin=[val_status_valid, val_status_invalid, val_status_needs_review]
)
grade_kwargs: dict[str, Any] = dict(
    isin=[VAL_GRADE_VALID, VAL_GRADE_INVALID, VAL_GRADE_NEEDS_REVIEW]
)
color_kwargs: dict[str, Any] = dict(
    isin=[VAL_COLOR_VALID, VAL_COLOR_INVALID, VAL_COLOR_NEEDS_REVIEW]
)


class FitAssessScores(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    w_type: str = w_type_field
    w_idx: int = w_idx_field
    time_start: float = pa.Field(**X_idx_field_kwargs)
    time_end: float = pa.Field(**X_idx_field_kwargs)
    area_mixed: float = pa.Field(**area_kwargs)
    area_unmixed: float = pa.Field(**area_kwargs)
    var_mixed: float = pa.Field(**var_kwargs)
    mean_mixed: float = pa.Field(**X_field_kwargs)
    fano_mixed: float = pa.Field(**fano_kwargs)
    score_recon: float = pa.Field(**score_kwargs)
    rtol: float = pa.Field(**rtol_kwargs)
    tolcheck: float = pa.Field(**tolcheck_kwargs)
    tolpass: bool = pa.Field(**tolpass_kwargs)
    fano_mean: float = pa.Field(**fano_mean_peaks_kwargs)
    div_fano: float = pa.Field(**div_fano_kwargs)
    fanopass: bool = pa.Field(**fanopass_kwargs)
    status: str = pa.Field(**status_kwargs)
    grade: str = pa.Field(**grade_kwargs)
    color: str = pa.Field(**color_kwargs)

    class Config:

        name = "FitAssessScores"
        ordered = True
        coerce = True
        strict = True


class PeakMsnts(pa.DataFrameModel):

    p_idx: int = mp_schs.p_idx_field_repeatable

    msnt: str = pa.Field(
        isin=[
            mp_defs.MAXIMA,
            mp_defs.KEY_WHH,
        ]
    )
    dim: str = pa.Field(
        isin=[
            mp_defs.X_IDX,
            mp_defs.X,
            mp_defs.KEY_WIDTH,
        ]
    )
    value: float

    class Config:
        strict = True
        ordered = True
        name = "Param Factory Peak Input"
        description = "Input peak parameters for DataPrepper"


class PeakMsntsWindowed(
    PeakMsnts,
    mw_schs.WindowFields,
):
    pass

    class Config:
        strict = True
        ordered = True
        name = "Windowed Param Factory Peak Input"
        description = "windowed form of OptParamPeakInput"


from dataclasses import dataclass
from pandera.typing import DataFrame


@dataclass
class DeconvolutionOutput:
    popt: DataFrame[Popt]
    psignals: DataFrame[PSignals]
    rsignal: DataFrame[RSignal]
    X_w_with_recon: DataFrame[X_Windowed_With_Recon]


class ActiveSignal(pa.DataFrameModel):
    w_type: str
    w_idx: int
    x: float
    mixed: float
    recon: float

    class Config:
        name = "Active Signal (Windowed)"
        

class ReconstructorSignalIn(pa.DataFrameModel):
    w_type: str
    w_idx: int
    x: float
    amplitude: float

    class Config:
        name = "Reconstructor Signal Input"


class ReconstructorPoptIn(pa.DataFrameModel):
    w_type: str
    w_idx: int
    maxima: float
    loc: float
    scale: float
    skew: float

    class Config:
        name = "Reconstructor Popt Table Input"


class TblSignalMixed(pa.DataFrameModel):
    w_type: str
    w_idx: int
    unit_idx: int
    x: float
    signal: str = pa.Field(isin=["X","X_corrected","recon"])
    amplitude: float
    
    class Config:
        name = "Table Mixed Signals"