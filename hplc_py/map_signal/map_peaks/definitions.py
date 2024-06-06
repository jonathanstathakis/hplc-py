from typing import TypedDict
from numpy.typing import ArrayLike
from hplc_py.common.definitions import X, IDX

PROMINENCE = 0.01
WLEN = None

KEY_PROMINENCE = "prominence"
KEY_WLEN = "wlen"

MAXIMA: str = "maxima"
P_IDX: str = "p_idx"
KEY_WIDTH_WHH: str = "width_whh"
KEY_WIDTH_PB: str = "width_pb"
KEY_LEFT_PROM: str = "left_prom"
KEY_RIGHT_PROM: str = "right_prom"
KEY_LEFT_WHH: str = "left_whh"
KEY_RIGHT_WHH: str = "right_whh"
KEY_WIDTH_WHH: str = "width_whh"
KEY_HEIGHT_WHH: str = "height_whh"
KEY_LEFT_PB: str = "left_pb"
KEY_RIGHT_PB: str = "right_pb"
KEY_WIDTH_PB: str = "width_pb"
KEY_HEIGHT_PB: str = "height_pb"

KEY_LEFT: str = "left"
KEY_RIGHT: str = "right"
KEY_PB: str = "pb"
KEY_PROM: str = "prom"
KEY_WHH: str = "whh"

KEY_IDX_ROUNDED: str = IDX + "_rounded"

KEY_MSNT = "msnt"
KEY_PEAK_PROP = "peak_prop"
LOC = "loc"
VALUE = "value"
DIM = "dim"

KEY_WIDTH = "width"


class MapPeaksKwargs(TypedDict, total=False):
    """
    kwargs for `scipy.signal.find_peaks`
    """

    height: float | ArrayLike
    threshold: float | ArrayLike
    distance: float
    prominence: float | ArrayLike
    width: float | ArrayLike
    wlen: float | None
    rel_height: float
    plateau_size: float | ArrayLike


map_peaks_kwargs_defaults = MapPeaksKwargs(
    wlen=None,
    prominence=0.01,
)
assert True