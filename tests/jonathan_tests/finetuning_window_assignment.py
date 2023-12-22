"""
2023-12-05 12:36:43

Investigate findwindows settings to produce an expected window range, that is:
- isolated peaks are covered by the window from left base to right base + buffer
- overlapping signals are considered in the same window, again, from left to right base
of the *region* + buffer.

TODO:
- [ ] in `baseline` rename the peak_df 'amp' column to 'amp_input', 'norm_amp' to 'amp_norm'

# Peak Prominence

Peak prominence is defined as the difference of the peak maxima and its lowest contour line.
 
Peak prominence calculation method:
1. Define window interval
    - extend a a horizontal line left or right of the peak maxima.
    - the extension stops either at a window bound ('wlen') or when the line encounters the slope again.
2. Define signal left and right bases.
    - find signal minima for the left and right window bound.
3. Calculate prominence
    - the higher of the left or right base is defined as the lowest contour line of the peak.
    - prominence is calculated as the difference between the peak maxima and its lowest contour line.
    
# Peak Widths

1. Evaluation height is calculated as peak maxima - peak prominence * rel_height. The more prominent the peak, the lower down the eval height.
2. draw a line at the evaluation height in both directions until:
    - the line intersects a slope
    - signal border
    - crosses the vertical position of the base
3. width is calculated as the distance between the endpoints defined in (2.). By definition the maximum width is the horizontal distance between the bases.

Currently, the width and the
"""

import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from hplc_py.quant import Chromatogram
import os

def overlay_windows(chm: Chromatogram,
                    signal_df,
                    window_df,
                    peak_signal_join,
                    ):
    
    fig, ax = plt.subplots(1)
    

    # a summary of the peak window regions
    pwtable = chm._findwindows.window_df_pivot(window_df)

    set2 = mpl.colormaps["Set2"].resampled(pwtable.groupby("window_idx").ngroups)
    
    # add the rectangles mapped to each window
    for id, window in pwtable.groupby("window_idx"):
        
        anchor_x = window["min"].values[0]
        anchor_y = 0
        width = window["max"].values[0] - window["min"].values[0]
        max_height = signal_df.norm_amp.max()
        
        rt = Rectangle(
            xy=(anchor_x, anchor_y),
            width=width,
            height=max_height,
            color=set2.colors[int(id) - 1],
        )

        ax.add_patch(rt)
    
    # do these last so they are on top of the rectangles
    # the signal slope        
    ax.plot(signal_df.time_idx, signal_df.norm_amp, label='signal')
    
    # the peak maxima
    ax.scatter(peak_signal_join.time_idx, peak_signal_join.norm_amp, label='peaks', color='red')
    
    plt.show()
    
    return None

def test_windowing(filepath: str):
    chm = Chromatogram()
    
    bcorr_outpath = os.path.join(os.getcwd(), "tests/jonathan_tests/bcorr_df.parquet")
    
    df = pd.read_csv(filepath).rename({'x':'time','y':'amp'},axis=1, errors='raise')
    df = chm.load_data(df) #type: ignore
    
    # amp_corrected, background = chm.baseline.correct_baseline(
    #     df.amp.to_numpy(np.float64),
    #     timestep=chm._dt,
    #     )
    # df = df.assign(amp_corrected = amp_corrected, background=background)
    
    # df.to_parquet(bcorr_outpath)
    
    df = pd.read_parquet(bcorr_outpath)
    signal_df, peak_df, window_df = chm._findwindows.profile_peaks_assign_windows(
        df.time.to_numpy(np.float64),
        df.amp_corrected.to_numpy(np.float64),
        chm._timestep.astype(npt.float64),
        )
    
    for df in [signal_df, peak_df, window_df]:
        print(df.head().to_markdown())
    
    pw_tbl = chm._findwindows.window_df_pivot(window_df)
    
    print(pw_tbl.to_markdown())
    
    peak_signal_join = (
        peak_df
            .set_index('time_idx')
         .join(
             signal_df,
             how='left'
         )
    )
    print(peak_signal_join.to_markdown())
    
    overlay_windows(
        chm,
        signal_df,
        window_df,
        peak_signal_join,
    )
    # so as we can see, in our version there are 4 very tight windows displayed.

    
    return None
    

def main():
    filepath = os.path.join(os.getcwd(),"tests/test_data/test_assessment_chrom.csv")
    test_windowing(filepath)

if __name__ == "__main__":
    main()