import pandas as pd
from cachier import cachier
import time

from hplc_py.common.caching import custom_param_hasher, CACHE_PATH

import os


@cachier(hash_func=custom_param_hasher, cache_dir=CACHE_PATH, allow_none=True)
def slow_func(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    arg_3: str,
    arg_4: int,
    arg_5: float,
    arg_6: list,
):
    print("Im being executed")
    time.sleep(1)
    # return None
