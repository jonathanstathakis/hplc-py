from typing import Optional

import pandas as pd
import pandera as pa
from numpy import float64, int64
from pandera.dtypes import String

from hplc_py.common_schemas import BaseDF
from hplc_py.common_schemas import HPLCBaseConfig




from ..common_schemas import (
    X_idx_field_max,
    X_idx_field_min,
    w_idx_field,
    w_type_field,
)

class Data(pa.DataFrameModel):
    """
    The central datastorage table of the Chromatogram object
    """

    w_type: Optional[String]
    w_idx: Optional[int64]
    t_idx: int64
    time: float64
    amp: float64
    amp_raw: Optional[float64]
    background: Optional[float64]

    class Config:
        name = "Data"
        coerce = True
        ordered = True
        strict = True


class RawData(BaseDF):
    """
    The base signal, with time and amplitude directions
    """

    t_idx: int64
    time: float64
    amp: float64

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "SignalDFLoaded"
        coerce = True


class SignalDFBCorr(Data):
    amp_raw: float64
    amp: float64

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "SignalDFBCorr"
        coerce = True


class FindPeaks(BaseDF):
    p_idx: int64 = pa.Field(ge=0, le=100, unique=True)
    X_idx: int64 = pa.Field(ge=0, le=5000, unique=True)
    maxima: float64 = pa.Field(ge=-1e-10, le=1000)
    prom: float64 = pa.Field(ge=0)
    prom_left: int64 = pa.Field(ge=X_idx_field_min, le=X_idx_field_max)
    prom_right: int64 = pa.Field(ge=X_idx_field_min, le=X_idx_field_max)

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "FindPeaks"
        coerce = True


class WHH(BaseDF):
    p_idx: int64
    whh_rel_height: pd.Float64Dtype
    whh_width: float64
    whh_height: float64
    whh_left: float64
    whh_right: float64

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WHH"
        coerce = True


class PeakBases(BaseDF):
    p_idx: int64
    pb_rel_height: float64
    pb_width: float64
    pb_height: float64
    pb_left: float64
    pb_right: float64

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PeakBases"
        coerce = True


class PeakMapWide(
    PeakBases,
    WHH,
    FindPeaks,
):
    class Config(HPLCBaseConfig):
        col_a_less_than_col_b = {"col_a": "whh_width", "col_b": "pb_width"}
        name = "PeakMapWide"


class PeakMapLong(pa.DataFrameModel):
    """
    Long form frame containing the concatenated peak property information indexed
    by peak number sorted by time
    """

    p_idx: int = pa.Field(ge=0)
    prop: str = pa.Field(
        isin=[
            "X_idx",
            "maxima",
            "prom",
            "prom_left",
            "prom_right",
            "whh_rel_height",
            "whh_width",
            "whh_height",
            "whh_left",
            "whh_right",
            "pb_rel_height",
            "pb_width",
            "pb_height",
            "pb_left",
            "pb_right",
        ]
    )
    value: float = pa.Field(ge=-0.1e-10)


class PeakMapWideColored(
    PeakBases,
    WHH,
    FindPeaks,
):
    color: object

    class Config(HPLCBaseConfig):
        col_a_less_than_col_b = {"col_a": "whh_width", "col_b": "pb_width"}
        name = "PeakMapWide"


class FitAssessScores(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    w_type: str = pa.Field()
    w_idx: int64 = pa.Field()
    time_start: float64 = pa.Field()
    time_end: float64 = pa.Field()
    signal_area: float64 = pa.Field()
    inferred_area: float64 = pa.Field()
    mixed_var: float64 = pa.Field()
    mixed_mean: float64 = pa.Field()
    mixed_fano: float64 = pa.Field()
    recon_score: float64 = pa.Field()
    rtol: float64 = pa.Field()
    tolcheck: float64 = pa.Field()
    tolpass: bool = pa.Field()
    u_peak_fano: float64 = pa.Field()
    fano_div: float64 = pa.Field()
    fanopass: bool = pa.Field()
    status: str = pa.Field()
    grade: str = pa.Field()
    color_tuple: str = pa.Field()

    class Config:

        name = "FitAssessScores"
        ordered = True
        coerce = True
        strict = True


class WdwPeakMapWide(PeakMapWide):
    w_type: pd.StringDtype = w_type_field
    w_idx: int64 = w_idx_field

    class Config:
        ordered = False
        strict = True

class ColorMap(pa.DataFrameModel):
    """
    A table mapping unique p_idxs to distinct colors. Use once to generate the mapping
    then join to the table as needed.
    """

    p_idx: int = pa.Field(ge=0)
    color: object
