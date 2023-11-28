import pandera as pa
import pandera.typing as pt

class WindowDF(pa.DataFrameModel):
    time_idx: pt.Series[int]
    window_id: pt.Series[int]
    window_type: pt.Series[str]
    
class WidthDF(pa.DataFrameModel):
        peak_idx: pt.Series[int]
        width: pt.Series[int]
        chl: pt.Series[float]
        left: pt.Series[float]
        right: pt.Series[float]