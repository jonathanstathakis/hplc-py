from typing import TypedDict, Optional
from numpy.typing import ArrayLike

class FindPeaksKwargs(TypedDict, total=False):
    height: Optional[float | ArrayLike]
    threshold: Optional[float | ArrayLike]
    distance: Optional[float]
    width: Optional[float | ArrayLike]
    plateau_size: Optional[float | ArrayLike]


class InterpModelKwargs(TypedDict, total=False):
    schema_name: str
    inherit_from: str
    is_base: bool
    check_dict: dict
    pandas_dtypes: bool
