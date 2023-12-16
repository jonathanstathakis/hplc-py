import pandas as pd
import numpy as np

df1 = pd.DataFrame(
    {
        'col1':np.arange(0,5,1),
        'col2':np.arange(7,12,1)
    }
)

dim1 = np.arange(5,10,1)
dim2 = np.arange(6,11,1)
array = np.asarray([[dim1,dim2]])

print(df1)
print(array)

df2 = df1.copy(deep=True)

df2 = array

print(df2)