from typing import Optional

import pandas as pd
import pandera as pa
from pandera.dtypes import String

from hplc_py.common.common_schemas import HPLCBaseConfig
from hplc_py.map_signal.map_peaks.schemas import PeakMap
from hplc_py.map_signal.map_windows.schemas import w_idx_field
from hplc_py.map_signal.map_windows.schemas import w_type_field


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


class SignalDFBCorr(Data):
    amp_raw: float
    amp: float

    class Config(HPLCBaseConfig):
        strict = True
        ordered = True
        name = "SignalDFBCorr"
        coerce = True


class WdwPeakMapWide(PeakMap):
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
