from numpy import float64, int64
from pandera.api.pandas.model_config import BaseConfig

from hplc_py import P0AMP, P0TIME, P0WIDTH, P0SKEW, AMPRAW, AMPCORR, AMP


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

    idx: Index[int] = pa.Field(check_name=True)

    @pa.check(
        "idx", name="idx_check", error="expected range index bounded by 0 and len(df)"
    )
    def check_is_range_index(cls, idx: Series[int]) -> bool:
        left_idx = pd.RangeIndex(0, len(idx) - 1)
        right_idx = pd.RangeIndex(idx.iloc[0], idx.iloc[-1])
        check = left_idx.equals(right_idx)
        return check

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "BaseDF"
        coerce = True


class SignalDFLoaded(BaseDF):
    """
    The base signal, with time and amplitude directions
    """
    time_idx: int64
    time: float64
    amp: float64

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "SignalDFLoaded"
        coerce = True
        


class SignalDFBCorr(pa.DataFrameModel):
    time: float64
    amp: float64
    amp_corrected: float64
    background: float64

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "SignalDFBCorr"
        coerce = True


class FindPeaks(BaseDF):
    p_idx: int64
    time_idx: int64
    time: float64
    amp: float64
    prom: float64
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
    whh_left_idx: float64
    whh_right_idx: float64
    whh_left_time: float64
    whh_right_time: float64
    
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
    pb_left_idx: float64
    pb_right_idx: float64
    pb_left_time: float64
    pb_right_time: float64
    
    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PeakBases"
        coerce = True


class PeakMap(
    PeakBases,
    WHH,
    FindPeaks,
):
    class Config(HPLCBaseConfig):
        col_a_less_than_col_b = {"col_a": "whh_width", "col_b": "pb_width"}
        name = "PeakMap"
        

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
    time_idx: int64 = pa.Field()
    time: float64 = pa.Field()
    amp_unmixed: float64 = pa.Field()

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PSignals"
        coerce = True


class RSignal(BaseDF):
    tform_state: str
    time_idx: int64
    time: float64
    amp: float64
    
    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "RSignal"
        coerce = True
        description = "The reconstituted signal, summation of the individual peak signals"


class PReport(Popt):
    retention_time: float64
    area_unmixed: float64
    maxima_unmixed: float64
    
    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PReport"
        coerce = True


class WindowedSignal(BaseDF):
    w_type: pd.StringDtype
    w_idx: int64
    time_idx: int64
    time: float64
    amp: float64

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowedSignal"
        coerce = True


class PeakWindows(BaseDF):
    time_idx: pd.Int64Dtype
    time: pd.Float64Dtype
    w_idx: pd.Int64Dtype
    w_type: pd.StringDtype

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PeakWindows"
        coerce = True

# 2024-01-29 10:50:49 JS: where is this used?
class IPBounds(BaseDF):
    ip_w_idx: int64
    ip_bound: pd.StringDtype
    time_idx: int64
    ip_w_type: pd.StringDtype

    @pa.check("ip_w_idx", name="check_w_idx_increasing")
    def check_monotonic_increasing_w_idx(cls, s: Series[int64]) -> bool:
        return s.is_monotonic_increasing

    @pa.check("time_idx", name="check_time_idx_increasing")
    def check_monotonic_increasing_t_idx(cls, s: Series[int64]) -> bool:
        return s.is_monotonic_increasing

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "IPBounds"
        coerce = True


class PWdwdTime(BaseDF):
    """
    peak windowed time dataframe, with NA's for nonpeak regions. An intermediate frame prior to full mapping
    """

    time_idx: int64
    time: float64
    w_idx: int64
    w_type: pd.StringDtype = pa.Field(isin=['peak','interpeak'])

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "PWdedTime"
        coerce = True
        

class WindowedTime(PWdwdTime):
    w_idx: int64 = pa.Field(gt=-1)
    
    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "WindowedTime"
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

    class Config:

        name="FitAssessScores"
        ordered=True
        coerce=True
        strict=True