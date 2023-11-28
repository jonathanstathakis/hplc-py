import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pt
import numpy.typing as npt
import matplotlib.pyplot as plt

from hplc_py.find_windows.find_windows_df_typing import WindowDF, WidthDF

class WindowFinderPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1)
    
    def plot_width_calc_overlay(self,
                                intensity: pt.Series,
                                peak_idx: pt.Series,
                                width_df: pt.DataFrame[WidthDF]
                                ):
        """
        For plotting the initial peak width calculations
        """
        # signal
        self.ax.plot(intensity, label='signal')
        # peak maxima
        self.ax.plot(peak_idx, intensity[peak_idx], '.', label='peak maxima')
        # widths measured at the countour line. widths are measured at 0.5, but then 
        # the left and right bases are returned for the user input value
        
        width_df.pipe(lambda x: plt.hlines(y= x['chl'],xmin=x['left'], xmax=x['right'], label='widths', color='orange'))
        self.ax.plot()
        self.ax.legend()
        
        return self.ax
    
    def plot_peak_ranges(self, ax, intensity:npt.NDArray[np.float64], ranges:list):
        """
        To be used to plot the result of WindowFinder.compute_peak_ranges
        """
        ax.plot(intensity, label='signal')
        for r in ranges:
            ax.plot(
                    r,
                    np.zeros(r.shape),
                    color='r')
        return ax