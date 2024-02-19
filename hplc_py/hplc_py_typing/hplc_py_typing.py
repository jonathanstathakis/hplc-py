from typing import Optional

import pandas as pd
import pandera as pa
from pandera.dtypes import String

from hplc_py.common_schemas import BaseDF
from hplc_py.common_schemas import HPLCBaseConfig
from hplc_py.map_peaks.schemas import PeakMapWide


from ..common_schemas import (
    w_idx_field,
    w_type_field,
)


class Data(pa.DataFrameModel):
    """
    The central datastorage table of the Chromatogram object
    """

    w_type: Optional[String]
    w_idx: Optional[int]
    t_idx: int
    time: float
    amp: float
    amp_raw: Optional[float]
    background: Optional[float]

    class Config:
        name = "Data"
        coerce = True
        ordered = True
        strict = True


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


class SignalDFBCorr(Data):
    amp_raw: float
    amp: float

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "SignalDFBCorr"
        coerce = True


class FitAssessScores(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    w_type: str = pa.Field()
    w_idx: int = pa.Field()
    time_start: float = pa.Field()
    time_end: float = pa.Field()
    signal_area: float = pa.Field()
    inferred_area: float = pa.Field()
    mixed_var: float = pa.Field()
    mixed_mean: float = pa.Field()
    mixed_fano: float = pa.Field()
    recon_score: float = pa.Field()
    rtol: float = pa.Field()
    tolcheck: float = pa.Field()
    tolpass: bool = pa.Field()
    u_peak_fano: float = pa.Field()
    fano_div: float = pa.Field()
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
    w_idx: int = w_idx_field

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
