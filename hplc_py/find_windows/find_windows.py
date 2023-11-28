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

from scipy import signal
import numpy as np
import warnings
import pandas as pd
import pandera as pa
import pandera.typing as pt
import pandera.typing as pt
import numpy.typing as npt
from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike
import matplotlib.pyplot as plt

from hplc_py.find_windows.find_windows_df_typing import WindowDF, WidthDF
from hplc_py.find_windows.find_windows_plot import WindowFinderPlotter

class WindowFinder:
    
    def __init__(self, viz:bool=True):
        self.__viz=viz
        
        if self.__viz:
            self.plotter = WindowFinderPlotter()
    
    def assign_windows(self,
                        time: pt.Series[float],
                        intensity: pt.Series[float],
                        timestep: float,
                        prominence:float=0.01,
                        rel_height:float=1,
                        buffer:int=0,
                        peak_kwargs:dict=dict())->pd.DataFrame:
    
        R"""
        Breaks the provided chromatogram down to windows of likely peaks. 

        Parameters
        ----------
        known_peaks : `list`
            The approximate locations of the peaks. If this is not provided, 
            peak locations will be automatically detected. 
        tolerance: `float`, optional
            If an enforced peak location is within tolerance of an automatically 
            identified peak, the automatically identified peak will be preferred. 
            This parameter is in units of time. Default is one-half time unit.
        prominence : `float`
            The promimence threshold for identifying peaks. Prominence is the 
            relative height of signal relative to the local background. 
        rel_height : `float`, [0, 1]
            The relative height of the peak where the baseline is determined. 
            Default is 1.
        buffer : positive `int`
            The padding of peak windows in units of number of time steps. Default 
            is 100 points on each side of the identified peak window.

        Returns
        ------- 
        window_df : `pandas.core.frame.DataFrame`
            A Pandas DataFrame with each measurement assigned to an identified 
            peak or overlapping peak set. This returns a copy of the chromatogram
            DataFrame with  a column  for the local baseline and one column for 
            the window IDs. Window ID of -1 corresponds to area not assigned to 
            any peaks
        """

        # input validation
        if (rel_height < 0) | (rel_height > 1):
            raise ValueError(f' `rel_height` must be [0, 1].')
        
        if not isArrayLike(time):
            raise TypeError(f"time must be array-like, got {type(intensity)}")

        if not isArrayLike(intensity):
            raise TypeError(f"intensity must be array-like, got {type(intensity)}")
        
    
        norm_int = self.normalize_intensity(intensity)
        
        width_df = self.construct_width_df(time,
                                        norm_int,
                                        prominence,
                                        rel_height,
                                        peak_kwargs
                                        )

        self.window_df = self.construct_window_df(
                                            norm_int=norm_int,
                                            time=time,
                                            width_df=width_df,
                                            buffer=buffer,
                                            )

        # Convert this to a dictionary for easy parsing
        window_dicts = {}
        
        for peak_window_id, peak_window_df in self.window_df.query("window_type == 'peak'").groupby('window_id'):
            if peak_window_id > 0:
                window_dicts[int(peak_window_id)] = self.create_window_dict(width_df['peak_idx'], width_df['width'], peak_window_df)

        
        self.window_props = window_dicts
        
        return window_df
    
    def compute_peak_idx(self, norm_int: pt.Series, prominence:float, peak_kwargs:dict=dict())->pt.Series:
            # Preform automated peak detection and set window ranges
            
            if not isArrayLike(norm_int):
                raise TypeError(f"norm_int should be array-like, got {type(norm_int)}")
    
            if norm_int.ndim!=1:
                raise ValueError('norm_int has too many dimensions, ensure is 1D array')
            
            if not len(norm_int)>1:
                raise ValueError(f'norm_int not long enough, got {len(norm_int)}')
            
            peak_idx, _ = signal.find_peaks(
                                norm_int,
                                prominence =prominence,
                                **peak_kwargs)
        
            return peak_idx

    def create_window_dict(self,
                           peak_idx: pt.Series,
                           widths: pt.Series,
                           peak_window: pt.DataFrame[WindowDF],
                           timestep: float
                           ):
        
        # filter peak_idx values if they are not also time_idx values (why?)
        _peak_idx = [p for p in peak_idx if p in peak_window['time_idx'].values]
                
        
        peak_inds = [x for _idx in _peak_idx for x in np.where(peak_idx == _idx)[0]]
        
        _dict = {
                        'time_range': peak_window['time'].values,
                         'signal': peak_window['signal'].values,
                         'signal_area': peak_window['signal'].values.sum(),
                         'num_peaks': len(_peak_idx),
                         'amplitude': [peak_window[peak_window['time_idx'] == idx]['signal'].values[0] for idx in _peak_idx],
                         
                         
                         'location': [peak_window[peak_window['time_idx'] == idx]['time'].values[0] for idx in _peak_idx],
                         
                         # convert widths from index units to time units
                         'width':  [widths[ind] * timestep for ind in peak_inds]}
        
        return _dict


    def mask_subset_ranges(self, ranges):
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
                          norm_int:pt.Series[float],
                          left_base:pt.Series[float],
                          right_base:pt.Series[float],
                          buffer:int=0,
                          ):
        """
        calculate the range of each peak based on the left and right base extended by 
        the buffer size, restricted to positive values and the length of the intensity
        array.
        
        Return a list of possible peak ranges
        """
        ranges = []
        
        for l, r in zip(left_base, right_base):
            
            peak_range = np.arange(int(l - buffer), int(r + buffer), 1)
            
            peak_range = peak_range[(peak_range >= 0) & (peak_range <= len(norm_int))]
            
            ranges.append(peak_range)
            
        return ranges
    
    def normalize_intensity(self, intensity: pt.Series[float])->pt.Series[float]:
        """
        Calculate and return the min-max normalized intensity, accounting for a negative baseline by extracting direction prior to normalization then reapplying before returning.
        """
        
        """
        if the range of values is [-5,-4,-3,-2,-1,0,1,2,3] then for say -1 the calculation is:
        
        y = (-1 - - 5)/(3 - - 5)
        y = (4)/(8)
        y = 1/2
        
        but if y was 1:
        
        y = (1 - - 5)/(3 - - 5)
        y = (6)/(8)
        y = 3/4
        
        and zero:
        
        y = (0 - - 5)/( 3 - - 5)
        y = 5/8
        
        The denominator stays the same regardless of the input. prior to the subtraction it reduces the numerator to:
        
        y = (x/8)-(min(x)/8)
        
        if x is negative, this becomes an addition, if positive, becomes a subtraction, contracting everything down to the range defined.
        
        for [-5,-4,-3,-2,-1,0,1,2,3]:
        
        int_sign = [-1, -1, -1, -1, -1, 0, 1, 1, 1]
        
        y = (x+5/8)
        and
                   [-5, -4,  -3,  -2,  -1,  0,  1,  2,  3]
        
        norm_int = [0, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1]
        
        multupling that by the negative sign would give you:
        
        [0, -1/8, -1/4, -3/8, -1/2, -5/8, +3/4, +7/8, +1]
        
        then multiplying again would simply reverse it.
        """
        
        if not isArrayLike(intensity):
            raise TypeError(f"intensity must be ArrayLike, got {intensity}")
        
        norm_int_magnitude = (intensity - intensity.min()) / \
            (intensity.max() - intensity.min())
        
        norm_int = norm_int_magnitude

        return norm_int

    def get_amps_inds(self, intensity: pt.Series, peak_idx: pt.Series)-> tuple:
        # Get the amplitudes and the indices of each peak
        peak_maxima_sign = np.sign(intensity[peak_idx])
        peak_maxima_pos = peak_maxima_sign > 0
        peak_maxima_neg = peak_maxima_sign < 0
        
        if not peak_maxima_sign.dtype==float:
            raise TypeError(f"peak_maximas_sign must be float, got {peak_maxima_sign.dtype}")
        if not peak_maxima_pos.dtype==bool:
            raise TypeError(f"peak_maxima_pos must be bool, got {peak_maxima_pos.dtype}")
        if not peak_maxima_neg.dtype==bool:
            raise TypeError(f"peak_maxima_pos must be bool, got {peak_maxima_neg.dtype}")
        
        return peak_maxima_sign, peak_maxima_pos, peak_maxima_neg
        
    def build_width_df(self,
                             intensity:pt.Series[float],
                             peak_idx:pt.Series[int],
                             rel_height:int|float=1,
                             )->pt.DataFrame[WidthDF]:
        """
        return the left and right bases of each peak as seperate arrays.
        """
        
        
        # Set up storage vectors for peak quantities
        
        width_df = pd.DataFrame(
            dict(
            peak_idx = peak_idx,
            width = np.zeros_like(len(peak_idx)),
            chl = 0,
            left=0,
            right=0
            )
        )
        
        # get the widths in index units at rel_height 0.5
        width_df['width'], _, _, _ = signal.peak_widths(intensity,
                                                        peak_idx,
                                                        rel_height=0.5)
        
        # contour line height
        _, width_df['clh'], width_df['left'], width_df['right'] = signal.peak_widths(intensity,
                                                                peak_idx,
                                                                rel_height=rel_height)
            
        
        return width_df
        
    def handle_background_windows(
        self,
        window_df: pt.DataFrame[WindowDF],
        bg_windows: pt.DataFrame[WindowDF],
        tidx
        ):
            split_inds = np.nonzero(
                    np.diff(bg_windows['time_idx'].values) - 1)[0]

                # If there is only one background window
            if (len(split_inds) == 0):
                window_df.loc[lambda df: df['time_idx'].isin(
                        bg_windows['time_idx'].values), 'window_id'] = 1
                window_df.loc[lambda df: df['time_idx'].isin(
                        bg_windows['time_idx'].values), 'window_type'] = 'interpeak'

                # If more than one split ind, set up all ranges.
            elif split_inds[0] != 0:
                split_inds += 1
                split_inds = np.insert(split_inds, 0, 0)
                split_inds = np.append(split_inds, len(tidx))

            bg_ranges = [bg_windows.iloc[np.arange(
                    split_inds[i], split_inds[i+1], 1)]['time_idx'].values for i in range(len(split_inds)-1)]
            win_idx = 1
            for i, rng in enumerate(bg_ranges):
                if len(rng) >= 10:
                    window_df.loc[window_df['time_idx'].isin(
                            rng), 'window_id'] = win_idx
                    window_df.loc[window_df['time_idx'].isin(
                            rng), 'window_type'] = 'interpeak'
                    win_idx += 1
                    
    def validate_ranges(self, ranges:list, mask:list)->list:
        
        validated_ranges = []
        for i, r in enumerate(ranges):
            if mask[i]:
                validated_ranges.append(r)
            
        return validated_ranges
        
    def construct_width_df(self,
                       time: pt.Series[float],
                       intensity: pt.Series[float],
                       prominence:float=0.01,
                       rel_height:float=1,
                       peak_kwargs:dict=dict()
                       )->pd.DataFrame:
        
        peak_idx = self.compute_peak_idx(intensity,
                                         prominence,
                                         peak_kwargs)
            
        width_df: pt.DataFrame[WidthDF] = self.build_width_df(
                                                  intensity,
                                                  peak_idx,
                                                  rel_height,
                                                  )    
        
        # create the widths plot for viz.
        if self.__viz:
            self.plotter.plot_width_calc_overlay(intensity, peak_idx, width_df)
        plt.show()
        
        return width_df
    
    def assign_window_id(self,
                        window_df: pt.DataFrame[WindowDF],
                        validated_ranges: list[list[int|float]],
                        )->pt.DataFrame:
        
        for i, r in enumerate(validated_ranges):
            
            # find 'time_idx' elements which correspond to the values of the range r
            mask = self.window_df['time_idx'].isin(r)
            
            # +1 because 0 is assigned to background windows
            window_df.loc[mask, 'window_id'] = int(i + 1)
        
        return window_df
    
    def construct_window_df(self,
                norm_int: pt.Series[float],
                time: pt.Series[float],
                width_df:pt.DataFrame[WidthDF],
                buffer:int=0,
                )->pd.DataFrame:
        
        window_df = pd.DataFrame(
            {
        "time_idx":np.arange(len(time)),
        "window_id":0,
        "window_type": 'peak',
            }
            )

        # calculate the range of each peak, returned as a list of ranges. Essentially
        # translates the calculated widths to time intervals, modified by the buffer
        ranges = self.compute_individual_peak_ranges(
                                        norm_int,
                                        width_df['left'],
                                        width_df['right'],
                                        buffer,
        )
    
        # Identiy subset ranges
        ranges_mask = self.mask_subset_ranges(ranges)

        # Keep only valid ranges and baselines        
        validated_ranges = self.validate_ranges(ranges, ranges_mask)
        
        # assign windows
        window_df = self.assign_window_id(window_df, validated_ranges)        

        # select the windows of the df not assigned to peaks
        bg_window_df = window_df.query("window_id==0")
        tidx = bg_window_df['time_idx'].values

        if len(bg_window_df) > 0:
        
            self.handle_background_windows(window_df, bg_window_df, tidx)
        
        # remove the background windows from the window_df
        window_df = window_df.query("window_id > 0")
        
        return window_df