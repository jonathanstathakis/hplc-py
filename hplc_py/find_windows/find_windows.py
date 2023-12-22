"""
1. Identify peaks in chromatographic data
2. clip the chromatogram into discrete peak windows

- Use `scipy.signal.find_peaks` with prominence as main filter

operation # 1 - find peaks
1. normalize before applying algorithm to generalise prominence filter settings
2. obtain the peak maxima indices

operation # 2 - clip the chromatogram into windows

- a window is defined as a region where peaks are overlapping or nearly overlapping.
- a window is identified by measuring the peak width at lowest contour line, peaks with 
overlapping contour lines are defined as influencing each other, therefore in the same window.
- buffer parameter used to control where windows start and finish, their magnitude (?)
"""

import typing
from scipy import signal #type: ignore
import numpy as np
import warnings
import pandas as pd
import pandera as pa
import pandera.typing as pt
import pandera.typing as pt
import numpy.typing as npt
from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

from hplc_py.hplc_py_typing.hplc_py_typing import (
                    OutSignalDF_Base,
                    OutPeakDF_Base,
                    OutWindowDF_Base,
                    )
from hplc_py.find_windows.find_windows_plot import WindowFinderPlotter

class WindowFinder:
    
    def __init__(self, viz:bool=True):
        self.__viz=viz
        
        if self.__viz:
            self.plotter = WindowFinderPlotter()
    
    def profile_peaks_assign_windows(self,
                        time: npt.NDArray[np.float64],
                        amp: npt.NDArray[np.float64],
                        prominence:float=0.01,
                        rel_height:float=1,
                        buffer:int=0,
                        peak_kwargs:dict=dict())->tuple[
                            pt.DataFrame[OutPeakDF_Base],
                            pt.DataFrame[OutWindowDF_Base],
                            ]:
    
        R"""
        Profile peaks and assign windows based on user input, return a dataframe containing
        time, signal, transformed signals, peak profile information
        """

        # input validation
        if (rel_height < 0) | (rel_height > 1):
            raise ValueError(f' `rel_height` must be [0, 1].')

        amp = np.asarray(amp, np.float64)
        time = np.asarray(time, np.float64)
        
        if amp.ndim!=1:
            raise ValueError("input amp must be 1D.")
        
        if time.ndim!=1:
            raise ValueError("input time must be 1D.")
        
        peak_df = self.peak_df_factory(
                                        time,
                                        amp,
                                        prominence,
                                        rel_height,
                                        peak_kwargs
                                        )

        window_df = self.window_df_factory(
                                            time,
                                            amp,
                                            peak_df['rl_left'].to_numpy(np.float64),
                                            peak_df['rl_right'].to_numpy(np.float64),
                                            buffer,
                                            )

        """
        normalize the data.
        
        get the window df as one which tracks which peaks are in which window and contains the metrics 'window area' and 'num peaks'
        
        Thus we end up with 4 tables:
        
        signal table: base and transformed signals.
        peak table: information on each peak.
        window metric table: window area, number of peaks and other metrics.
        
        keys:
        window metric table: window id
        peak table: peak maxima time index, foreign key: window id
        signal table: time idx
        
        key table:
         - window id
         - peak id
         - time idx
        """
        return peak_df, window_df
    
    def normalize_series(self, x: npt.NDArray[np.float64])->npt.NDArray[np.float64]:
        """
        Calculate and return the min-max normalized intensity, accounting for a negative baseline by extracting direction prior to normalization then reapplying before returning.
        """
        
        amp_norm = (x - x.min()) / \
            (x.max() - x.min())
    
        return amp_norm
    
    def norm_inverse(
        self,
        x: npt.NDArray[np.float64],
        x_norm: npt.NDArray[np.float64]
    ):
        """
        Invert the normalization to express the measure heights in base scale
        """
        
        x_inv = x_norm * (x.max()-x.min())+x.min()
        
        return x_inv
        
    
    def peak_df_factory(self,
                         time: npt.NDArray[np.float64],
                         amp: npt.NDArray[np.float64],
                         prominence:float=0.01,
                         rel_height: float=1,
                         peak_kwargs:dict=dict(),
                         )->pt.DataFrame:
        
        # Preform automated peak detection and set window ranges
        
        time = np.asarray(time, dtype=np.float64)
        amp = np.asarray(amp, dtype=np.float64)
        
        prominence=float(prominence)
        rel_height=float(rel_height)
        peak_kwargs=dict(peak_kwargs)

        if len(amp)<1:
            raise ValueError(f'input to,e not long enough, got {len(amp)}')
        if len(time)<1:
            raise ValueError(f'input amplitude not long enough, got {len(amp)}')
        
        if time.ndim!=1:
            raise ValueError('time has too many dimensions, ensure is 1D array')
        if amp.ndim!=1:
            raise ValueError('input amplitude has too many dimensions, ensure is 1D array')
        
        # normalize amplitude
        
        amp_norm = self.normalize_series(amp)
        
        time_idx, _ = signal.find_peaks(
                            amp_norm,
                            prominence =prominence,
                            **peak_kwargs)
        
        time_idx = np.asarray(time_idx, np.int64)
        
        if len(time_idx)<1:
            raise ValueError("length of 'time_idx' is less than 1")
        
        peak_prom, _, _ = signal.peak_prominences(amp_norm,
                                            time_idx,)
        
        peak_prom = np.asarray(peak_prom, np.float64)
        if len(peak_prom)<1:
            raise ValueError("length of 'peak_prom' is less than 1")

        # width is calculated by first identifying a height to measure the width at, calculated as:
        # (peak height) - ((peak prominance) * (relative height))
        
        # width half height, width half height height
        # measure the width at half the hieght for a better approximation of
        # the latent peak
        
        # this measurement defines the 'scale' paramter of the skewnorm distribution
        # for the signal peak reconstruction
        
        whh, whhh, whh_left, whh_right = signal.peak_widths(
            amp_norm,
            time_idx,
            rel_height=0.5
        )
        
        # get the left and right bases of a peak measured at user defined location
        
        # rel_height width, width height, left and right
        # the values returned by the user specified 'rel_height', defaults to 1.
        
        # this measurement defines the windows
        
        rl_width, rl_wh, rl_left, rl_right = signal.peak_widths(
                                                            amp_norm,
                                                            time_idx,
                                                            rel_height=rel_height)
        
        # convert wh measurements to base scale
        
        whhh_inv = self.norm_inverse(amp, whhh)
        rl_wh_inv = self.norm_inverse(amp, rl_wh)
        
        whh = np.asarray(whh, np.float64)
        whhh_inv = np.asarray(whhh_inv, np.float64)
        whh_left = np.asarray(whh_left, np.float64)
        whh_right = np.asarray(whh_right, np.float64)
        
        rl_width = np.asarray(rl_width, np.float64)
        rl_wh_inv = np.asarray(rl_wh_inv, np.float64)
        rl_left = np.asarray(rl_left, np.float64)
        rl_right = np.asarray(rl_right, np.float64)
        
        
        if len(whh)<1:
            raise ValueError("length of 'whh' is less than 1")
        if len(whhh_inv)<1:
            raise ValueError("length of 'whhh' is less than 1")
        if len(whh_left)<1:
            raise ValueError("length of 'whh_left' is less than 1")
        if len(whh_right)<1:
            raise ValueError("length of 'whh_right' is less than 1")
        
        if len(rl_width)<1:
            raise ValueError("length of 'width' is less than 1")
        if len(rl_wh_inv)<1:
            raise ValueError("length of 'width_height' is less than 1")
        if len(rl_left)<1:
            raise ValueError("length of 'left' is less than 1")
        if len(rl_right)<1:
            raise ValueError("length of 'right' is less than 1")
    
    
        peak_df = (pd.DataFrame(
        {
        'time_idx': pd.Series(time_idx, dtype=pd.Int64Dtype()),
        'peak_prom': pd.Series(peak_prom, dtype=pd.Float64Dtype()),
        'whh': pd.Series(whh, dtype=pd.Float64Dtype()),
        'whhh': pd.Series(whhh_inv, dtype=pd.Float64Dtype()),
        'whh_left': pd.Series(whh_left, dtype=pd.Float64Dtype()),
        'whh_right': pd.Series(whh_right, dtype=pd.Float64Dtype()),
        'rel_height': pd.Series([rel_height]*len(time_idx), dtype=pd.Float64Dtype()),
        'rl_width': pd.Series(rl_width, dtype=pd.Float64Dtype()),
        'rl_wh': pd.Series(rl_wh_inv, dtype=pd.Float64Dtype()),
        'rl_left': pd.Series(rl_left, dtype=pd.Float64Dtype()),
        'rl_right': pd.Series(rl_right, dtype=pd.Float64Dtype()),
        },
        )
        .rename_axis('peak_idx')
        .reset_index()
        )
        

        return typing.cast(pt.DataFrame[OutPeakDF_Base],peak_df)

    def mask_subset_ranges(self, ranges:list[npt.NDArray[np.int64]])->npt.NDArray[np.bool_]:
        """
        Generate a boolean mask of the peak ranges array which defaults to True, but
        False if for range i, range i+1 is a subset of range i.
        """
        # generate an array of True values
        valid = np.full(len(ranges), True)
        
        '''
        if there is more than one range in ranges, set up two identical nested loops.
        The inner loop skips the first iteration then checks if the values in range i+1
        are a subset of range i, if they are, range i+1 is marked as invalid. 
        A subset is defined as one where the entirety of the subset is contained within
        the superset.
        
        
        '''
        if len(ranges) > 1:
            for i, r1 in enumerate(ranges):
                for j, r2 in enumerate(ranges):
                    if i != j:
                        if set(r2).issubset(r1):
                            valid[j] = False
        return valid

    def compute_individual_peak_ranges(self,
                          amp:npt.NDArray[np.float64],
                          left_base:npt.NDArray[np.float64],
                          right_base:npt.NDArray[np.float64],
                          buffer:int=0,
                          )->list[npt.NDArray[np.int64]]:
        """
        calculate the range of each peak based on the left and right base extended by 
        the buffer size, restricted to positive values and the length of the intensity
        array.
        
        Return a list of possible peak ranges
        """
        
        left_base = np.array(left_base,np.float64)
        right_base = np.array(right_base,np.float64)
        
        if len(left_base)<1:
            raise ValueError("left base must be longer than 1")
        if len(right_base)<1:
            raise ValueError("right base must be longer than 1")
        
        if left_base.ndim>1:
            raise ValueError("left_base must be a 1d array")
        if right_base.ndim>1:
            raise ValueError("right_base must be a 1d array")
        ranges = []
        
        for l, r in zip(left_base, right_base):
            
            peak_range = np.arange(int(l - buffer), int(r + buffer), 1)
            
            # retrict ranges to between 0 and the end of the signal
            peak_range = peak_range[(peak_range >= 0) & (peak_range <= len(amp))]
            
            ranges.append(peak_range)
            
        return ranges

    def get_amps_inds(self, intensity: pt.Series, time_idx: pt.Series)-> tuple:
        # Get the amplitudes and the indices of each peak
        peak_maxima_sign = np.sign(intensity[time_idx])
        peak_maxima_pos = peak_maxima_sign > 0
        peak_maxima_neg = peak_maxima_sign < 0
        
        if not peak_maxima_sign.dtype==float:
            raise TypeError(f"peak_maximas_sign must be float, got {peak_maxima_sign.dtype}")
        if not peak_maxima_pos.dtype==bool:
            raise TypeError(f"peak_maxima_pos must be bool, got {peak_maxima_pos.dtype}")
        if not peak_maxima_neg.dtype==bool:
            raise TypeError(f"peak_maxima_pos must be bool, got {peak_maxima_neg.dtype}")
        
        return peak_maxima_sign, peak_maxima_pos, peak_maxima_neg

    
    def compute_peak_time_ranges(self,
                                 norm_int: npt.NDArray[np.float64],
                                 left: npt.NDArray[np.float64],
                                 right: npt.NDArray[np.float64],
                                 buffer: int,
                                 )->list[npt.NDArray[np.int64]]:
        """
        calculate the range of each peak, returned as a list of ranges. Essentially translates the calculated widths to time intervals, modified by the buffer. Background ranges are defined implicitely as the regions of the time idex not covered by any range.
        """
        
        norm_int = np.asarray(norm_int, dtype=np.float64)
        left = np.asarray(left,dtype=np.int64)
        right = np.asarray(right,dtype=np.int64)
        
        if len(norm_int)==0:
            raise ValueError("amplitude array has length 0")
        if len(left)==0:
            raise ValueError("left index array has length 0")
        if len(right)==0:
            raise ValueError("right index array has length 0")
        
        ranges = self.compute_individual_peak_ranges(
                                        norm_int,
                                        left,
                                        right,
                                        buffer,
        )
    
        # Identiy subset ranges
        ranges_mask = self.mask_subset_ranges(ranges)

        # Keep only valid ranges and baselines        
        
        validated_ranges = []
        for i, r in enumerate(ranges):
            if ranges_mask[i]:
                validated_ranges.append(r)
        
        if len(validated_ranges)==0:
            raise ValueError("Something has gone wrong with the ranges or the validation")
                
        return validated_ranges
        
    
    def window_df_factory(self,
                time: npt.NDArray[np.float64],
                amp: npt.NDArray[np.float64],
                left_indices: npt.NDArray[np.float64],
                right_indices: npt.NDArray[np.float64],
                buffer:int=0,
                )->pt.DataFrame[OutWindowDF_Base]:
        
        time = np.asarray(time, np.float64)
        amp = np.asarray(amp, np.float64)
        left_indices = np.asarray(left_indices, np.float64)
        right_indices = np.asarray(right_indices, np.float64)
        
        if len(amp)==0:
            raise ValueError("amplitude array has length 0")
        if len(left_indices)==0:
            raise ValueError("left index array has length 0")
        if len(right_indices)==0:
            raise ValueError("right index array has length 0")
        
        peak_windows = self.compute_peak_time_ranges(
            amp,
            left_indices,
            right_indices,
            buffer
        )
        
        # now stack the frame so that the column labels become a second column and
        # the values are arranged vertically. Add a value of 1 to the 'window_idx' to
        # reserve 0 for background regions.
        # Because the ranges are of uneven length, NAs are introduced when they are formed into a DataFrame. Melting them propagates the NAs througout the 'range_idx' column. Thus dropping NAs should be appropriate.
        window_df = (pd.DataFrame(peak_windows, dtype=pd.Int64Dtype())
                     .T
                     .rename_axis(columns='window_idx')
                     .rename_axis(index='i')
                     .melt(value_name='time_idx')
                     .dropna()
                     .reset_index(drop=True)
                     .assign(**{'window_idx':lambda df: df['window_idx']+1})
                     )
        

        # add a binary 'window_type' column reflecting whether the row is a peak or not
        mask = (window_df['window_idx']==0)
        window_df['window_type']=np.where(mask,'interpeak', 'peak')
        
        return typing.cast(pt.DataFrame[OutWindowDF_Base],window_df)
    
    @pa.check_types
    def window_df_pivot(self,
                        window_df: pt.DataFrame[OutWindowDF_Base],
                        time_col: str='time_idx',
                        window_idx_col: str='window_idx',
                        aggfuncs:list=['min','max'],
                        )-> pd.DataFrame:
        """
        pivot the window df to neatly provide a description of each window
        
        TODO:

        - [ ] establish schema of the pivot table
        """
        
        
        pivot_window_df = (
            window_df.pivot_table(
                values=time_col, columns=window_idx_col, aggfunc=aggfuncs,
            )
            .stack(1)
            .reset_index()
        )
        
        return pivot_window_df
    
    @pa.check_types
    def display_windows(
        self,
        peak_df: pt.DataFrame[OutPeakDF_Base],
        signal_df: pt.DataFrame[OutSignalDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
        time_col: str='time_idx',
        y_col: str='amp_corrected',
        ax=None,
        
    ):
        if not ax:
            fig, _ax = plt.subplots(1)
        else:
            _ax=ax

        peak_signal_join = (
            peak_df.set_index(time_col)
            .join(signal_df.set_index(time_col), how="left", validate="1:1")
            .reset_index()
        )

        # the signal

        _ax.plot(signal_df[time_col], signal_df[y_col], label="signal")

        pwtable = self.window_df_pivot(window_df)

        def signal_window_overlay(
            ax,
            signal_df: pt.DataFrame[OutSignalDF_Base],
            pwtable: pd.DataFrame,
            window_idx_col: str='window_idx',
            y_col: str='amp_corrected',
            
        ) -> None:
            """
            Create an overlay of the signal and the windows.
            """

            set2 = mpl.colormaps["Set2"].resampled(
                pwtable.groupby(window_idx_col).ngroups
            )

            for id, window in pwtable.groupby(window_idx_col):
                anchor_x = window["min"].values[0]
                anchor_y = 0
                width = window["max"].values[0] - window["min"].values[0]
                max_height = signal_df[y_col].max()

                rt = Rectangle(
                    xy=(anchor_x, anchor_y),
                    width=width,
                    height=max_height,
                    color=set2.colors[int(id) - 1],  # type: ignore
                )

                ax.add_patch(rt)

            return ax

        signal_window_overlay(_ax,
                            signal_df,
                            pwtable
                            )

        # the peaks

        _ax.scatter(
            peak_signal_join[time_col],
            peak_signal_join[y_col],
            label="peaks",
            color="red",
        )

        # now draw the interpolations determining the peak width
        if not ax:
            fig.show() #type: ignore
            plt.show()
        else:
            return _ax
            