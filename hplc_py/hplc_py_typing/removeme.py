import pandas as pd
import numpy as np
from hplc_py.hplc_py_typing.hplc_py_typing import checkArrayLike
d = [1,2,3,4,5]
s = pd.Series(d, index=np.arange(len(d)), name='data')

print(checkArrayLike(s))

