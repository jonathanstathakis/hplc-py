import os
import pandas as pd
import numpy as np
from hplc_py.quant import Chromatogram
import matplotlib.pyplot as plt
import seaborn as sns

def get_parquet_data(filepath:str):
    data = pd.read_parquet(filepath)
    return data



def main():

    path = "/Users/jonathan/hplc-py/tests/jonathan_workshop/oriada111.parquet"
    
    # my data
    
    md = get_parquet_data(path)
    
    md = (md
            # .query("(time>0)&(time<5)")
    )
    
    chm = Chromatogram(md)
    
    # # test data
    # # chm = Chromatogram(file=testdata, cols={'time':'x','signal':'y'}) 
    
    chm.correct_baseline(window=0.5)
    chm._assign_windows(
        buffer=1
        )
    
    chm.fit_peaks(approx_peak_width=0.5, buffer=1)

    
    
    chm.show()
    
    plt.show()
    
    # chm.show()
    
if __name__ == "__main__":
    main()