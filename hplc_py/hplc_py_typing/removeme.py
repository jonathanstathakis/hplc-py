import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike
d = [1,2,3,4,5]
s = Series(d, index=np.arange(len(d)), name='data')

print(isArrayLike(s))

