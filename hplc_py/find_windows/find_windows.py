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
from numpy.typing import ArrayLike
from hplc_py.hplc_py_typing.hplc_py_typing import isArrayLike
import matplotlib.pyplot as plt

class WindowFinderPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1)
    
    def plot_width_calc_overlay(self, intensity, peak_idx, width_df):
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
    
    def plot_peak_ranges(self, ax, intensity:ArrayLike, ranges:list):
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
    

class WindowFinder:
    
    def __init__(self, viz:bool=True):
        self.window_df = pd.DataFrame()
        self.__viz=viz
        
        if self.__viz:
            self.plotter = WindowFinderPlotter()
    
    def _assign_windows(self,
                        time: ArrayLike,
                        intensity: ArrayLike,
                        known_peaks=[],
                        tolerance=0.5,
                        prominence=0.01,
                        rel_height=1,
                        buffer=0,
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
        
    def assign_windows(self,
                       time: ArrayLike,
                       intensity: ArrayLike,
                       buffer:int=0,
                       prominence: float=0.01,
                       rel_height: float=1,
                       peak_kwargs:dict=dict()
                       ):
        
        norm_int = self.normalize_intensity(intensity)
        width_df = self.identify_peaks(time,
                                       norm_int,
                                       prominence,
                                       rel_height,
                                       peak_kwargs
                                       )
    
        # calculate the range of each peak, returned as a list of ranges. Essentially
        # translates the calculated widths to time intervals, modified by the buffer
        ranges = self.compute_peak_ranges(
                                        norm_int,
                                        width_df['left'],
                                        width_df['right'],
                                        buffer,
        )
    
        # Identiy subset ranges
        ranges_mask = self.mask_subset_ranges(ranges)

        # Keep only valid ranges and baselines        
        self.ranges = self.validate_ranges(ranges, ranges_mask)
        
        assert False
        
        # assign windows
        
        for i, r in enumerate(ranges):
            self.window_df.loc[self.window_df['time_idx'].isin(r),
                          'window_id'] = int(i + 1)

        # Determine the windows for the background (nonpeak) areas.
        bg_windows = self.window_df[self.window_df['window_id'] == 0]
        tidx = bg_windows['time_idx'].values

        if len(bg_windows) > 0:
            self.handle_background_windows(self.window_df, bg_windows, tidx)
        self.window_df = self.window_df[self.window_df['window_id'] > 0]

        # Convert this to a dictionary for easy parsing
        window_dicts = {}
        for peak_window_id, peak_window in self.window_df[self.window_df['window_type'] == 'peak'].groupby('window_id'):
            if peak_window_id > 0:
                _dict = self.create_window_dict(peak_idx, width_df['width'], peak_window)
                window_dicts[int(peak_window_id)] = _dict

        
        self.window_props = window_dicts
        
        return self.window_df
    
    def compute_peak_idx(self, norm_int: ArrayLike, prominence:float, peak_kwargs:dict=dict())->ArrayLike:
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

    def create_window_dict(self, peak_idx, _widths, peak_window):
        _peak_idx = [
                    p for p in peak_idx if p in peak_window['time_idx'].values]
                
        assert len(_peak_idx)>0
                
        peak_inds = [x for _idx in _peak_idx for x in np.where(
                    peak_idx == _idx)[0]]
        
        _dict = {'time_range': peak_window[self.time_col].values,
                         'signal': peak_window[self.int_col].values,
                         'signal_area': peak_window[self.int_col].values.sum(),
                         'num_peaks': len(_peak_idx),
                         'amplitude': [peak_window[peak_window['time_idx'] == idx][self.int_col].values[0] for idx in _peak_idx],
                         'location': [peak_window[peak_window['time_idx'] == idx][self.time_col].values[0] for idx in _peak_idx],
                         'width':  [_widths[ind] * self._dt for ind in peak_inds]}
        
        return _dict
    
    def setup_window_df(self, length: int):
        
        window_df = pd.DataFrame(
            dict(
        time_idx= np.arange(length),
        window_id=0,
        window_type= 'peak',
            )
        )
        return window_df

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

    def compute_peak_ranges(self,
                          norm_int:ArrayLike,
                          left_base:ArrayLike,
                          right_base:ArrayLike,
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
    
    def normalize_intensity(self, intensity: ArrayLike)->ArrayLike:
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

    def get_amps_inds(self, intensity: ArrayLike, peak_idx: ArrayLike)-> tuple:
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
    
    def get_peak_width_props(self,
                             intensity:ArrayLike,
                             peak_idx:ArrayLike,
                             rel_height:int|float=1,
                             )->tuple:
        """
        return the left and right bases of each peak as seperate arrays.
        """
        
        
        # Set up storage vectors for peak quantities
        l = len(peak_idx
                )
        
        width_df = pd.DataFrame(
            dict(
            peak_idx = peak_idx,
            width = np.zeros_like(l),
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
        
    def handle_background_windows(self, window_df, bg_windows, tidx):
            split_inds = np.nonzero(
                    np.diff(bg_windows['time_idx'].values) - 1)[0]

                # If there is only one background window
            if (len(split_inds) == 0):
                window_df.loc[window_df['time_idx'].isin(
                        bg_windows['time_idx'].values), 'window_id'] = 1
                window_df.loc[window_df['time_idx'].isin(
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
        
    def identify_peaks(self, time: ArrayLike,
                       intensity: ArrayLike,
                       prominence:float=0.01,
                       rel_height:float=1,
                       peak_kwargs:dict=dict(),
                       ):
        # setup
        self.window_df = self.setup_window_df(len(time))
        
        peak_idx = self.compute_peak_idx(intensity,
                                         prominence,
                                         peak_kwargs)
        
        # Get the peak properties
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=signal._peak_finding_utils.PeakPropertyWarning)
            
            
        width_df = self.get_peak_width_props(
                                                  intensity,
                                                  peak_idx,
                                                  rel_height,
                                                  )    
        
        # create the widths plot for viz.
        if self.__viz:
            self.plotter.plot_width_calc_overlay(intensity, peak_idx, width_df)
        plt.show()
        
        return width_df