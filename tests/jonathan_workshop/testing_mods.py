import os
import pandas as pd
import numpy as np
from hplc_py.quant import Chromatogram
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

plt.rcParams['figure.dpi']=140
plt.rcParams['figure.figsize']=(5,4)

def get_parquet_data(filepath:str):
    data = pd.read_parquet(filepath)
    return data

def sharpen_signal(signal: pd.Series, k: float=1):
    
    sharpened_signal = signal-k*signal.diff().diff()
    
    sharpened_signal.loc[0:1]=signal.loc[0:1]
    
    return sharpened_signal
def main():

    path = "/Users/jonathan/hplc-py/tests/jonathan_workshop/oriada111.parquet"
    
    # my data
    
    md = get_parquet_data(path)
    
    md = (md
            # .query("(time>0)&(time<5)")
    )
    
    md['sharpened'] = sharpen_signal(md.signal, 1.1)
    
    
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2,2)
    
    # subplot title font size
    sptfs = 8
    
    timesnapped_md = md.query(f"(time>=2)&(time<=4.5)")
    axs[0][0].plot(timesnapped_md.time, timesnapped_md.signal, label='raw signal')
    axs[0][0].plot(timesnapped_md.time, timesnapped_md.sharpened, label='sharpened')
    axs[0][0].set_title("raw and sharpend t>=2 & t<=4.", fontsize=sptfs)
    
    axs[0][1].plot(md.time, md.signal, label='raw signal')
    axs[0][1].plot(md.time, md.sharpened, label='sharpened')
    axs[0][1].set_title('raw and sharpened', fontsize = sptfs)
    
    timesnapped_md_2 = md.query("(time<2)")
    axs[1][0].plot(timesnapped_md_2.time, timesnapped_md_2.signal, label='raw signal')
    axs[1][0].plot(timesnapped_md_2.time, timesnapped_md_2.sharpened, label='sharpened')
    axs[1][0].set_title('raw and sharpened t < 2', fontsize=sptfs)
    
    # fig.tight_layout()
    # test data
    # chm = Chromatogram(file=testdata, cols={'time':'x','signal':'y'}) 
    
    raw_signal = md.signal
    
    md['smoothed'] = signal.savgol_filter(md.signal, window_length=5, polyorder=2)
    
    axs[1][1].plot(md.time, md.sharpened, label='sharpened')
    axs[1][1].plot(md.time, md.smoothed, label = 'smoothed')
    axs[1][1].set_title('sharpened and smoothed', fontsize = sptfs)
    md = md.query("time>2")
    
    chm = Chromatogram(
    md,
    cols=dict(time='time',signal='sharpened')
    )
    
    '''
    
    '''
    
    peaks = chm.fit_peaks(
        approx_peak_width=0.7,
        )

    def create_subplots(chm: Chromatogram):
        fig, axs = plt.subplots(2,3)
        
        time_bins = pd.cut(md.time, 5).drop_duplicates()
        
        for i, ax in enumerate(axs.flatten()):
            if i == 5:
                break
            tr = time_bins.iloc[i]
            # ax.plot([1,2,3],[4,5,6])
            chm.show(fig=fig,ax=ax  , time_range=[tr.left, tr.right])
        
        # chm.show()
        fig.delaxes(axs.flatten()[-1])
        # fig.legend(loc='lower right')
        fig.tight_layout()
        plt.show()
        
    create_subplots(chm)

    
if __name__ == "__main__":
    main()