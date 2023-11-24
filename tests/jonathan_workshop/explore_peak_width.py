import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def subset_data(df: pd.DataFrame):
    
    df = df.query("(time<=4.5)&(time>=2)")
    return df

def sharpen_data(df: pd.DataFrame):
    
    return df

def find_peak_profiles(df: pd.DataFrame, ):
    """
    find_peak_profiles return the peak idx, amplitudes, widths, and left and right interpolated intersection points

    _extended_summary_

    :param df: _description_
    :type df: pd.DataFrame
    :return: _description_
    :rtype: _type_
    """
    
    x = df['signal']
    
    timestep = df.time.diff().mean()
    
    print(timestep)
    
    prominence = x.max()*0.01
    
    peak_idxs, _ = signal.find_peaks(x, prominence=prominence)
    
    peak_loc = df.iloc[peak_idxs]['time']
    peak_amps = x.iloc[peak_idxs]
    
    peak_widths, width_heights, left_ips, right_ips =  signal.peak_widths(x, peaks=peak_idxs, rel_height=1)
    
    # find the closest times to the ips
    
    peaks_df = pd.DataFrame(
        dict(
            peak_idx = peak_idxs,
            peak_loc = peak_loc,
            peak_amp = peak_amps,
            peak_widths=peak_widths*timestep,
            width_heights=width_heights,
            left_ips=left_ips*timestep+df.time.min(),
            right_ips=right_ips*timestep+df.time.min(),
        )
    )
    
    return peaks_df

def plot_results(signal_df: pd.DataFrame, peak_df: pd.DataFrame)->None:
    """
    Create a plot of the result of the peak widths calculations superimposed on the signal. Draw a horizontal line from left_ips to right_ips at width_heights, label the peaks etc.
    """
    
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(1)
    
    ax.plot(signal_df.time, signal_df.signal, label='signal')
    ax.scatter(peak_df.peak_loc, peak_df.peak_amp, label='peaks', marker='.', color='red', s=100)
    
    
    
    ax.hlines(*[peak_df.width_heights, peak_df.left_ips, peak_df.right_ips])
    
    [10,]
    
    plt.show()
    return None

def main():
    
    path = "/Users/jonathan/hplc-py/tests/jonathan_workshop/oriada111.parquet"
    
    md = pd.read_parquet(path)
    sd = subset_data(md)
    ssd = sharpen_data(sd)
    peak_df = find_peak_profiles(ssd)
    
    display(peak_df)
    plot_results(ssd, peak_df)

if __name__ == "__main__":
    main()