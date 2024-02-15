from pandera.dtypes import UInt64
from pandera.dtypes import String
import os
import warnings
from hplc_py import SCHEMA_CACHE
from typing import Optional

from numpy import float64, int64
from pandera.api.pandas.model_config import BaseConfig

from hplc_py import P0AMP, P0TIME, P0WIDTH, P0SKEW, AMPRAW, AMPCORR, AMP
from hplc_py.hplc_py_typing.custom_checks import col_a_less_than_col_b

import pandas as pd
import pandera as pa
from pandera.typing import Index, Series


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
    p_idx: int64 = pa.Field(ge=0, le=100)
    X_idx: int64 = pa.Field(ge=0, le=5000)
    maxima: float64 = pa.Field(ge=-1e-10, le=1000)
    prom: float64 = pa.Field(ge=0)
    prom_left: int64
    prom_right: int64

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


class P0(BaseDF):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    w_idx: int64 = pa.Field()
    p_idx: int64 = pa.Field()
    param: pd.CategoricalDtype = pa.Field(isin=[P0AMP, P0TIME, P0WIDTH, P0SKEW])
    p0: float64 = pa.Field()

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "P0"
        coerce = True


class Bounds(BaseDF):
    w_idx: int64
    p_idx: int64
    param: pd.CategoricalDtype
    lb: float64 = pa.Field(nullable=False)
    ub: float64 = pa.Field(nullable=False)

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

    w_idx: int64 = pa.Field()
    p_idx: int64 = pa.Field()
    amp: float64 = pa.Field()
    time: float64 = pa.Field()
    whh_half: float64 = pa.Field()
    skew: float64 = pa.Field()

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "Popt"
        coerce = True


class PSignals(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    p_idx: int64 = pa.Field()
    t_idx: int64 = pa.Field()
    time: float64 = pa.Field()
    amp_unmixed: float64 = pa.Field()

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PSignals"
        coerce = True


class RSignal(BaseDF):
    t_idx: int64
    time: float64
    amp_unmixed: float64

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "RSignal"
        coerce = True
        description = (
            "The reconstituted signal, summation of the individual peak signals"
        )


class PReport(Popt):
    retention_time: float64
    area_unmixed: float64
    maxima_unmixed: float64

    class Config(HPLCBaseConfig):
        strict = True
        ordered = False
        name = "PReport"
        coerce = True



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


class X_Schema(pa.DataFrameModel):
    X_idx: int
    X: float64

    class Config:
        strict = True
        description = "A simplistic container for the signal array"
        unique=['X_idx']


class WdwPeakMapWide(PeakMapWide):
    w_type: pd.StringDtype
    w_idx: int64

    class Config:
        ordered = False
        strict = True
        unique=['X_idx', 'p_idx']


class InP0(pa.DataFrameModel):
    w_idx: int64
    p_idx: int64
    amp: float64
    time: float64
    whh: float64 = pa.Field(alias="whh_width")

    class Config(BaseConfig):
        name = "in_p0"
        ordered = False
        strict = True


class ColorMap(pa.DataFrameModel):
    """
    A table mapping unique p_idxs to distinct colors. Use once to generate the mapping
    then join to the table as needed.
    """

    p_idx: int = pa.Field(ge=0)
    color: object
