import pandas as pd
import numpy as np
from hplc_py.quant import Chromatogram

def get_data(filepath:str):
    data = pd.read_parquet(filepath)
    return data

def main():

    oriada111 = get_data("./tests/jonathan_workshop/oriada111.parquet")
    
    chm = Chromatogram(oriada111)
    
    chm.fit_peaks(approx_peak_width=0.7)
    
    chm.show()
    
if __name__ == "__main__":
    main()