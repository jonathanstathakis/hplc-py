"""
Pandera schema classes for pipelines in `map_peaks_viz` module.
"""
import pandera as pa


class PM_Width_In_X(pa.DataFrameModel):
    p_idx: int  # the peak idx
    whh_left: float
    whh_right: float
    pb_left: float
    pb_right: float

    class Config:
        name = "PM_WIdth_In_X"
        description = "the x values of the peak map data"
        strict = True


class PM_Width_In_Y(pa.DataFrameModel):
    p_idx: int  # the peak idx
    whh_height: float
    pb_height: float

    class Config:
        name = "PM_Width_In_Y"
        description = "the y values of the widths"
        strict = True


class PM_Width_Long_Out_X(pa.DataFrameModel):
    """
    A generalized schema to verify that the x frame is as expected after
    melting.
    """

    p_idx: int = pa.Field(ge=0)  # peak idx
    peak_prop: str = pa.Field(isin=["whh", "pb"])
    geoprop: str = pa.Field(isin=["left", "right"])
    x: float

    class Config:
        name = "PM_Width_Long_Out_X"
        strict = True


class PM_Width_Long_Out_Y(pa.DataFrameModel):
    """
    A generalized schema to verify that the y frames are as expected after
    melting.
    """

    p_idx: int = pa.Field(
        ge=0,
    )  # peak idx
    peak_prop: str = pa.Field(isin=["whh", "pb"])
    geoprop: str = pa.Field(isin=["height"])
    y: float

    class Config:
        name = "PM_Width_Long_Out_Y"
        strict = True


class PM_Width_Long_Joined(pa.DataFrameModel):
    p_idx: int = pa.Field(
        ge=0,
    )
    peak_prop: str = pa.Field(
        isin=["whh", "pb"],
    )
    geoprop: str = pa.Field(isin=["left", "right"])
    x: float
    y: float

    class Config:
        name = "PM_Width_Long_Joined"
        strict = True


class Maxima_X_Y(pa.DataFrameModel):
    """
    Schema for a 3 column frame respresnting each peaks maxima x and y indexed
    by peak index.
    """

    p_idx: int = pa.Field(ge=0)
    maxima_x: int = pa.Field(gt=0)
    maxima_y: float

    class Config:
        name = "Maxima_X_Y"
        strict = True

class Width_Maxima_Join(pa.DataFrameModel):
    """
    Schema for the width, maxima join ready for plotting lines tracing from the width
    ips to peak maxima
    """
    
    p_idx: int = pa.Field(ge=0)
    peak_prop: str = pa.Field(isin=["whh","pb"])
    geoprop: str = pa.Field(isin=["left","right"])
    x1: float
    y1: float
    x2: int
    y2: float
    
    class Config:
        name = "Width_Maxima_Join"
        strict = True
        ordered=True