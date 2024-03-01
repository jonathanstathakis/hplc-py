from typing import TypedDict, Optional
from numpy.typing import ArrayLike

class InterpModelKwargs(TypedDict, total=False):
    schema_name: str
    inherit_from: str
    is_base: bool
    check_dict: dict
    pandas_dtypes: bool
