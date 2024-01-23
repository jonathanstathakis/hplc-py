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

    def plot_signal(
        self,
        signal_df: pd.DataFrame,
        time_col: str,
        amp_col: str,
        ax: plt.Axes,
    ):
        x = signal_df[time_col]
        y = signal_df[amp_col]
        
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
        amp_unmixed = (unmixed_df
              .pivot_table(columns='p_idx', values='amp_unmixed',index='time')
              .sum(axis=1)
              .reset_index()
              .rename({0:"amp_unmixed"},axis=1)
              )
        x = amp_unmixed['time']
        y = amp_unmixed['amp_unmixed']
        
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
            p_idx = df.loc[0,'p_idx']
            
            x = df['time']
            y = df['amp_unmixed']
            
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            pc = ax.fill_between(x,
                            y,
                            alpha=0.5)
            
            facecolor = pc.get_facecolor()
            # see https://matplotlib.org/stable/users/explain/axes/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
            patch = mpatches.Patch(color=facecolor, label=p_idx)
            ax.legend(handles=[patch])
            
            # ax.annotate(df['p_idx'][0], xy=[])
            return ax
        
        ax = unmixed_df.groupby('p_idx').apply(plot_peak, ax)
    
        return ax