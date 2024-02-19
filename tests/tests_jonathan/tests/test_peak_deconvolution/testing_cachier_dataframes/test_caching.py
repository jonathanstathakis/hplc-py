"""
2024-02-19 18:11:43

developing a method of persistent caching of pandas dataframes. Static Frame provides an implementation in the docs which i will test here
"""

import pandas as pd
import time
import os
from hplc_py import ROOT
import static_frame as sf
from typing import Callable
from .slow_funcs import slow_func

columns = pd.Index([0, 1])
rows = [
    ["a", 1],
    ["b", 2],
    ["c", 3],
]
df_1 = pd.DataFrame(rows, columns=columns).astype({0: pd.StringDtype(), 1: int})
df_2 = df_1.assign(c=["rabbit", "ogre", "roof"])

start = time.process_time()
slow_func(
    df_1=df_1,
    df_2=df_2,
    arg_3="b",
    arg_4=1,
    arg_5=5.0,
    arg_6=["a"]
    ) # fmt: skip
end = time.process_time()
elapsed_time = end - start
print(elapsed_time)
