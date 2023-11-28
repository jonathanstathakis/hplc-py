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

from hplc_py.hplc_py_typing.hplc_py_typing import WindowedSignalDF, WidthDF
from hplc_py.find_windows.find_windows_plot import WindowFinderPlotter

class WindowFinder:
    
    def __init__(self, viz:bool=True):
        self.__viz=viz
        
        if self.__viz:
            self.plotter = WindowFinderPlotter()
    
    def assign_windows(self,
                        amp: npt.NDArray[np.float64],
                        time: npt.NDArray[np.float64],
                        timestep: npt.NDArray[np.float64],
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

        amp = np.asarray(amp, np.float64)
        time = np.asarray(time, np.float64)
        
        if amp.ndim!=1:
            raise ValueError("input amp must be 1D.")
        
        if time.ndim!=1:
            raise ValueError("input time must be 1D.")
        
        norm_amp = self.normalize_intensity(amp)
        
        width_df = self.construct_width_df(
                                        norm_amp,
                                        prominence,
                                        rel_height,
                                        peak_kwargs
                                        )

        window_df = self.window_signal_df(
                                            norm_amp,
                                            time,
                                            width_df.left.to_numpy(np.float64),
                                            width_df.right.to_numpy(np.float64),
                                            buffer,
                                            )

        
        # Convert this to a dictionary for easy parsing
        self.window_props = self.construct_dict_of_window_dicts(window_df,
                                                                width_df,
                                                                window_df,
                                                                timestep
                                                                )
        
        return window_df
    
    def compute_peak_idx(self,
                         norm_int: npt.NDArray[np.float64],
                         prominence:float, peak_kwargs:dict=dict()
                         )->npt.NDArray[np.int64]:
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
        
            return peak_idx.astype(np.int64)

    def create_window_dict(self,
                           peak_idx: pt.Series[int],
                           widths: pt.Series[float],
                           peak_window_df: pt.DataFrame[WindowedSignalDF],
                           timestep: float
                           ):
        
        # filter peak_idx values if they are not also time_idx values (why?)
        _peak_idx = [p for p in peak_idx if p in peak_window_df['time_idx'].values]
                
        
        peak_inds = [x for _idx in _peak_idx for x in np.where(peak_idx == _idx)[0]]
        
        _dict = {
                        'time_range': peak_window_df['time'].values,
                         'signal': peak_window_df['signal'].values,
                         'signal_area': peak_window_df['signal'].values.sum(),
                         'num_peaks': len(_peak_idx),
                         'amplitude': [peak_window_df[peak_window_df['time_idx'] == idx]['signal'].values[0] for idx in _peak_idx],
                         
                         
                         'location': [peak_window_df[peak_window_df['time_idx'] == idx]['time'].values[0] for idx in _peak_idx],
                         
                         # convert widths from index units to time units
                         'width':  [widths[ind] * timestep for ind in peak_inds]}
        
        return _dict


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
                          norm_int:npt.NDArray[np.float64],
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
        ranges = []
        
        for l, r in zip(left_base, right_base):
            
            peak_range = np.arange(int(l - buffer), int(r + buffer), 1)
            
            peak_range = peak_range[(peak_range >= 0) & (peak_range <= len(norm_int))]
            
            ranges.append(peak_range)
            
        return ranges
    
    def normalize_intensity(self, amp: npt.NDArray[np.float64])->npt.NDArray[np.float64]:
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
        
        if not isArrayLike(amp):
            raise TypeError(f"intensity must be ArrayLike, got {amp}")
        
        norm_int_magnitude = (amp - amp.min()) / \
            (amp.max() - amp.min())
        
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
                             peak_idx:npt.NDArray[np.int64],
                             amp:npt.NDArray[np.float64],
                             rel_height:int|float=1,
                             )->pt.DataFrame[WidthDF]:
        """
        return the left and right bases of each peak as seperate arrays.
        """
        
        
        # Set up storage vectors for peak quantities
        
        width_df = pd.DataFrame(
            {
            'peak_idx': peak_idx,
            'width': np.zeros_like(len(peak_idx)),
            'clh': 0,
            'left': 0,
            'right': 0,
            }
        )
        
        # get the widths in index units at rel_height 0.5
        width_df['width'], _, _, _ = signal.peak_widths(amp,
                                                        peak_idx,
                                                        rel_height=0.5)
        
        # contour line height
        _, width_df['clh'], width_df['left'], width_df['right'] = signal.peak_widths(amp,
                                                                peak_idx,
                                                                rel_height=rel_height)
            
        
        return width_df.pipe(pt.DataFrame[WidthDF])
        
    def label_background_windows(
        self,
        window_df: pt.DataFrame[WindowedSignalDF],
        ):
        
        bg_window_df = window_df.query("window_id==0")
        
        # find the indices of the bounds of the background regions. The discrete
        # difference of two neighbouring integers will always be one. Therefore if there
        # is a jump in indice values due to the presence of a peak, the diff will be
        # larger. Subtracting 1 from the diff array will produce an array with a number
        # of zeroes, which can be filtered with `np.nonzero`. `np.nonzero' can handle
        # n dimensions, with each element of the return tuple for each dimension. As
        # we are working with 1D array, just return the first element`
        
        tidx:npt.NDArray[np.int64] = bg_window_df['time_idx'].values
        tidx_diff:npt.NDArray[np.int64] = np.diff(tidx)
        split_inds: npt.NDArray[np.int64] = np.nonzero(tidx_diff - 1)[0]

        # below is assigning 'interpeak' to window_df to label background regions (?)
        
        # If there is only one background window dont need complicated logic
        if len(split_inds) == 0:
            
            mask = window_df['time_idx'].isin(bg_window_df['time_idx'])
            window_df.loc[mask, 'window_id'] = 1 # type: ignore
            window_df.loc[mask, 'window_type'] = 'interpeak' # type: ignore

        # If more than one split ind, set up all ranges.
        elif split_inds[0] != 0:
            
            # shift all the background idx values up by 1
            split_inds += 1
            
            # insert a zero value at the first position
            split_inds = np.insert(split_inds, 0, 0)
            
            # insert the length of the time index as the final value of the array
            split_inds = np.append(split_inds, len(tidx))
        
        background_intervals = []
        
        for i in range(len(split_inds)-1):
            
            # form pairs of 'split_inds', the indices indicating the bounds of the 
            # background regions
            split_ind_pair = np.arange(split_inds[i], split_inds[i+1], 1)
            
            # get the time index value for the corresponding indices
            background_interval = bg_window_df.iloc[split_ind_pair]['time_idx'].values
            
            # add them to the list of intervals
            background_intervals.append(background_interval)
        
        win_idx = 1
        
        for bg_range in background_intervals:
            
            # not worth it for ranges smaller than 10? so you'd have an implicitely
            # defined region that is neither background or peak..
            
            if len(bg_range) >= 10:
                
                # find time index values for corresponding background range
                bg_mask = window_df['time_idx'].isin(
                        bg_range)
                
                # assign 'window_id' of 1 for background ranges
                window_df.loc[bg_mask, 'window_id'] = win_idx
                
                # assign 'window_type' as 'interpeak'
                window_df.loc[bg_mask, 'window_type'] = 'interpeak'
                
                win_idx += 1
        
        return window_df
                    
    def validate_ranges(self, ranges:list[npt.NDArray[np.int64]], mask:npt.NDArray[np.bool_])->list[npt.NDArray[np.int64]]:
        
        validated_ranges = []
        for i, r in enumerate(ranges):
            if mask[i]:
                validated_ranges.append(r)
            
        return validated_ranges
        
    def construct_width_df(self,
                       amp: npt.NDArray[np.float64],
                       prominence:float=0.01,
                       rel_height:float=1,
                       peak_kwargs:dict=dict()
                       )->pt.DataFrame[WidthDF]:
        
        peak_idx = self.compute_peak_idx(amp,
                                         prominence,
                                         peak_kwargs)
            
        width_df = self.build_width_df(
                            peak_idx,
                            amp,
                            rel_height,
                            )    
        
        # create the widths plot for viz.
        if self.__viz:
            self.plotter.plot_width_calc_overlay(
                amp,
                peak_idx,
                width_df)
            plt.show()
        
        return width_df.pipe(pt.DataFrame[WidthDF])
    
    def build_windowed_signal_df(self,
                         time: npt.NDArray[np.float64],
                         amp: npt.NDArray[np.float64],
                        validated_ranges: list[npt.NDArray[np.int64]],
                        )->pt.DataFrame:
        
        window_df = pd.DataFrame(
            {
        "time_idx":np.arange(len(time)),
        "time":time,
        "amp":amp,
        "window_id":0,
        "window_type": 'peak',
            }
            )
        
        for i, r in enumerate(validated_ranges):
            
            # find 'time_idx' elements which correspond to the values of the range r
            mask = window_df['time_idx'].isin(r)
            
            # +1 because 0 is assigned to background windows
            window_df.loc[mask, 'window_id'] = int(i + 1)
        
        return window_df.pipe(pt.DataFrame[WindowedSignalDF])
    
    def window_signal_df(self,
                amp: npt.NDArray[np.float64],
                time: npt.NDArray[np.float64],
                left:npt.NDArray[np.float64],
                right:npt.NDArray[np.float64],
                buffer:int=0,
                )->pt.DataFrame[WindowedSignalDF]:

        ranges = self.compute_peak_time_ranges(
            amp,
            left,
            right,
            buffer
        )
        
        windowed_signal_df = self.build_windowed_signal_df(
            time,
            amp,
            ranges
            ) 
        
        if 0 in windowed_signal_df['window_id'].values:
            self.label_background_windows(windowed_signal_df)
        
        # remove the background windows from the window_df
        windowed_signal_df = windowed_signal_df.query("window_id > 0")
        
        return windowed_signal_df.pipe(pt.DataFrame[WindowedSignalDF])

    def compute_peak_time_ranges(self,
                                 norm_int: npt.NDArray[np.float64],
                                 left: npt.NDArray[np.float64],
                                 right: npt.NDArray[np.float64],
                                 buffer: int,
                                 )->list[npt.NDArray[np.int64]]:
        # calculate the range of each peak, returned as a list of ranges. Essentially
        # translates the calculated widths to time intervals, modified by the buffer
        ranges = self.compute_individual_peak_ranges(
                                        norm_int,
                                        left,
                                        right,
                                        buffer,
        )
    
        # Identiy subset ranges
        ranges_mask = self.mask_subset_ranges(ranges)

        # Keep only valid ranges and baselines        
        validated_ranges = self.validate_ranges(ranges, ranges_mask)
                
        return validated_ranges
        
        
    def construct_dict_of_window_dicts(
        self,
        window_df: pt.DataFrame[WindowedSignalDF],
        width_df:pt.DataFrame[WidthDF],
        peak_window_df,
        timestep: float,
        )->dict[dict]:
        
        window_dicts = {}
        
        for peak_window_id, peak_window_df in window_df.query("window_type == 'peak' & window_id>0").groupby('window_id'):
                
                window_dicts[peak_window_id] = self.create_window_dict(
                    width_df['peak_idx'],
                    width_df['width'],
                    peak_window_df,
                    timestep
                    )
        return window_dicts