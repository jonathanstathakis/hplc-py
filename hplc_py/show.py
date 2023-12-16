import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import numpy.typing as npt
import pandera as pa
import pandera.typing as pt


"""
Module for vizualisation, primarily the "Show" class
"""

class Show:
    '''
    - [x] plot the raw chromatogram
    - [x] plot the reconstructed signal as 'inferred mixture'
    - [x] plot each peak and fill between
    - [ ] plot estimated background
    - [ ] set up a plot if none provided
    '''
    def __init__(self):
        pass

    def plot_raw_chromatogram(
        self,
        signal_df,
        ax,
    ):
        print(signal_df.head())
        x = signal_df['time_idx']
        y = signal_df['amp']
        
        ax.plot(
         x,y, label='bc chromatogram'   
        )
        return ax
        
    def plot_reconstructed_signal(
        self,
        unmixed_df,
        ax,
    ):
        '''
        Plot the reconstructed signal as the sum of the deconvolved peak series
        '''
        unmixed_amp = (unmixed_df
              .pivot_table(columns='peak_idx', values='unmixed_amp',index='time_idx')
              .sum(axis=1)
              .reset_index()
              .rename({0:"unmixed_amp"},axis=1)
              )
        x = unmixed_amp['time_idx']
        y = unmixed_amp['unmixed_amp']
        
        ax.plot(x,y, label='reconstructed signal')
        
        return ax

    def plot_individual_peaks(
        self,
        unmixed_df,
        ax,
    ):
        '''
        Plot the individual deconvolved peaks with a semi-transparent fill to demonstrate overlap
        '''
        def plot_peak(df, ax):
            peak_idx = df.loc[0,'peak_idx']
            
            x = df['time_idx']
            y = df['unmixed_amp']
            
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            pc = ax.fill_between(x,
                            y,
                            alpha=0.5)
            
            facecolor = pc.get_facecolor()
            # see https://matplotlib.org/stable/users/explain/axes/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
            patch = mpatches.Patch(color=facecolor, label=peak_idx)
            ax.legend(handles=[patch])
            
            # ax.annotate(df['peak_idx'][0], xy=[])
            return ax
        
        ax = unmixed_df.groupby('peak_idx').apply(plot_peak, ax)
    
        return ax