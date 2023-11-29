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

TODO:

- ratify each test to each function now that cleanup is done
- define test schemas corresponding to the input
- change DF initialization to initializtion ON values rather than before - implicit casting to input datatype makes initialization prior to assignment useless.
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

from hplc_py.hplc_py_typing.hplc_py_typing import BaseWindowedSignalDF, PeakDF, BaseAugmentedDF, AugmentedDataFrameWidthMetrics
from hplc_py.find_windows.find_windows_plot import WindowFinderPlotter

class WindowFinder:
    
    def __init__(self, viz:bool=True):
        self.__viz=viz
        
        if self.__viz:
            self.plotter = WindowFinderPlotter()
    
    def profile_peaks_assign_windows(self,
                        time: npt.NDArray[np.float64],
                        amp: npt.NDArray[np.float64],
                        timestep: float,
                        prominence:float=0.01,
                        rel_height:float=1,
                        buffer:int=0,
                        peak_kwargs:dict=dict())->pt.DataFrame[AugmentedDataFrameWidthMetrics]:
    
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
        
        norm_amp = self.normalize_intensity(amp)
        
        if ((np.max(norm_amp)!=1) | (np.min(norm_amp)!=0)):
            raise ValueError("'norm_amp' does not range between 0 and 1")
        
        peak_df = self.build_peak_df(
                                        norm_amp,
                                        prominence,
                                        rel_height,
                                        peak_kwargs
                                        )

        windowed_signal_df = self.window_signal_df(
                                            amp,
                                            norm_amp,
                                            time,
                                            peak_df['rl_left'].to_numpy(np.float64),
                                            peak_df['rl_right'].to_numpy(np.float64),
                                            buffer,
                                            )

        
        # Convert this to a dictionary for easy parsing
        augmented_df = self.create_augmented_df(
                                            peak_df,
                                            windowed_signal_df,
                                            timestep
                                            )
        
        return augmented_df.pipe(pt.DataFrame[AugmentedDataFrameWidthMetrics])
    
    def build_peak_df(self,
                         amp: npt.NDArray[np.float64],
                         prominence:float=0.01,
                         rel_height: float=1,
                         peak_kwargs:dict=dict(),
                         )->npt.NDArray[np.int64]:
        # Preform automated peak detection and set window ranges
        
        amp = np.asarray(amp, dtype=np.float64)
        
        prominence=float(prominence)
        rel_height=float(rel_height)
        peak_kwargs=dict(peak_kwargs)

        if len(amp)<1:
            raise ValueError(f'input amplitude not long enough, got {len(amp)}')
        
        if amp.ndim!=1:
            raise ValueError('norm_int has too many dimensions, ensure is 1D array')

        
        maxima_idx, _ = signal.find_peaks(
                            amp,
                            prominence =prominence,
                            **peak_kwargs)
        
        maxima_idx = np.asarray(maxima_idx, np.int64)
        
        if len(maxima_idx)<1:
            raise ValueError("length of 'maxima_idx' is less than 1")
        
        peak_prom, _, _ = signal.peak_prominences(amp,
                                            maxima_idx,)
        
        peak_prom = np.asarray(peak_prom, np.float64)
        if len(peak_prom)<1:
            raise ValueError("length of 'peak_prom' is less than 1")

        # width is calculated by first identifying a height to measure the width at, calculated as:
        # (peak height) - ((peak prominance) * (relative height))
        
        # width half height, width half height height
        # measure the width at half the hieght for a better approximation of
        # the latent peak
        whh, whhh, whh_left, whh_right = signal.peak_widths(
            amp,
            maxima_idx,
            rel_height=0.5
        )
        
        # get the left and right bases of a peak measured at user defined location
        
        # rel_height width, width height, left and right
        # the values returned by the user specified 'rel_height', defaults to 1.
        rl_width, rl_wh, rl_left, rl_right = signal.peak_widths(
                                                            amp,
                                                            maxima_idx,
                                                            rel_height=rel_height)
        
        
        
        whh = np.asarray(whh, np.int64)
        whhh = np.asarray(whhh, np.int64)
        whh_left = np.asarray(whh_left, np.int64)
        whh_right = np.asarray(whh_right, np.int64)
        
        rl_width = np.asarray(rl_width, np.int64)
        rl_wh = np.asarray(rl_wh, np.int64)
        rl_left = np.asarray(rl_left, np.int64)
        rl_right = np.asarray(rl_right, np.int64)
        
        
        if len(whh)<1:
            raise ValueError("length of 'whh' is less than 1")
        if len(whhh)<1:
            raise ValueError("length of 'whhh' is less than 1")
        if len(whh_left)<1:
            raise ValueError("length of 'whh_left' is less than 1")
        if len(whh_right)<1:
            raise ValueError("length of 'whh_right' is less than 1")
        
        if len(rl_width)<1:
            raise ValueError("length of 'width' is less than 1")
        if len(rl_wh)<1:
            raise ValueError("length of 'width_height' is less than 1")
        if len(rl_left)<1:
            raise ValueError("length of 'left' is less than 1")
        if len(rl_right)<1:
            raise ValueError("length of 'right' is less than 1")
        
        
        peak_df = pd.DataFrame(
        {
        'maxima_idx': pd.Series(maxima_idx, dtype=pd.Int64Dtype()),
        'peak_prom': pd.Series(peak_prom, dtype=pd.Float64Dtype()),
        'whh': pd.Series(whh, dtype=pd.Float64Dtype()),
        'whhh': pd.Series(whhh, dtype=pd.Int64Dtype()),
        'whh_left': pd.Series(whh_left, dtype=pd.Int64Dtype()),
        'whh_right': pd.Series(whh_right, dtype=pd.Int64Dtype()),
        'rl_width': pd.Series(whh, dtype=pd.Float64Dtype()),
        'rl_wh': pd.Series(whhh, dtype=pd.Int64Dtype()),
        'rl_left': pd.Series(whh_left, dtype=pd.Int64Dtype()),
        'rl_right': pd.Series(whh_right, dtype=pd.Int64Dtype()),
        },
        )

        return peak_df.pipe(pt.DataFrame[PeakDF])
    
    

    def create_augmented_df(self,
                           peak_df: pt.DataFrame[PeakDF],
                           windowed_signal_df: pt.DataFrame[BaseWindowedSignalDF],
                           timestep: np.float64
                           )->pt.DataFrame:
        """
        for each subset of windowed_signal_df break them down into dicts.
        
        How about no?
        
        ~time_range = time~
        ~signal=signal~
        signal_area = signal.sum()
        num_peaks = len(maxima_idx)
        amplitude = peak maxima amplitude
        location = peak maxima time value
        width = width in time units (multiplied by timestep)
        
        return it as a dataframe with the following values.
        """
        
        if not peak_df['maxima_idx'].isin(windowed_signal_df['time_idx']).any():
            raise ValueError("peak df 'maxima_idx' value(s) not in 'time_idx'")
        
        # join windowed_signal_df and width_df on time_idx, maxima_idx
        
        # from merge docs NaN in either side will match each other, thus raise an
        # error if present to avoid unexpected behavior
        
        if windowed_signal_df['time_idx'].isna().sum()>0:
            raise ValueError("NaN present in join key column")
        
        if peak_df['maxima_idx'].isna().sum()>0:
            raise ValueError("NaN present in join key column")    

        # for window id i am expecting a corresponding range index which contains the idx
        # of the maxima. On that I can merge the peak_df such that the values are smeared
        # across the entire range. call it maxima index.
        # to do that: find the window id for each maxima_idx. groupby window id, assign maxima index if maxima index in
        # group.
        
        maxima_idx = peak_df.maxima_idx
        
        l_idx_df = len(windowed_signal_df)
        # add maxima_idx to idx_df
        merged_df = pd.merge(
            windowed_signal_df,
            maxima_idx,
            left_on='range_idx',
            right_on='maxima_idx',
            how='left'
            )
        
        if l_idx_df != len(merged_df):
            raise RuntimeError("idx_df has unexpectedly changed length after merge. check inputs")
        # now smear
        
        merged_df = (
            merged_df
            .assign(maxima_idx=lambda df: 
                  df
                  .groupby('window_id')['maxima_idx']
                  .fillna(method='ffill')
                )
            .assign(maxima_idx=lambda df: 
                  df
                  .groupby('window_id')['maxima_idx']
                  .fillna(method='bfill')
                  )
        )

        # Test if 'window_id' and 'maxima_idx' are aligned. if not, the gradient of 'maxima_idx' in each group will not equal zero.
        
        gradient = merged_df.groupby('window_id')['maxima_idx'].diff().fillna(0)
        
        if (
            gradient!=0).any():
            raise ValueError(
                "window_id and maxima_idx are not aligned at index:"
                f"{gradient[gradient!=0]}"
                )
        
        # now we can use maxima_idx to join the rest of the table.
        
        try:
            augmented_df = (pd.merge(merged_df, peak_df,
                                    left_on='maxima_idx',
                                    right_on='maxima_idx',
                                    how='left',
                                    validate='many_to_one',
                                    )
                                .astype({
                                    'maxima_idx':pd.Int64Dtype(),
                                })
            )
        
        except Exception as e:
            raise RuntimeError(e)
            
        try:
             augmented_df = augmented_df.pipe(pt.DataFrame[BaseAugmentedDF])
        except Exception as e:
            raise RuntimeError(f"schema failed - {e}")
        
        augmented_df=augmented_df.assign(**{
            'window_area':pd.Series([], dtype=pd.Float64Dtype()),
            'num_peaks':pd.Series([], dtype=pd.Int64Dtype()),
        }).pipe(pt.DataFrame[AugmentedDataFrameWidthMetrics])
        
        # seperate then merge back.
        
        # need to perform the calculations and assignment in one sweep. or not
        
        def add_metrics(df: pd.DataFrame):
            
            # window AUC
            df['window_area'] = df['amp'].sum()
            df['num_peaks']=df['maxima_idx'].nunique()
            df['amplitude']=df['amp'].max()
            
            return df
        
        augmented_df = augmented_df.groupby('window_id', as_index=False).apply(add_metrics)

        return augmented_df.pipe(pt.DataFrame[AugmentedDataFrameWidthMetrics])


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
        
        norm_int = (amp - amp.min()) / \
            (amp.max() - amp.min())
        
        return norm_int

    def get_amps_inds(self, intensity: pt.Series, maxima_idx: pt.Series)-> tuple:
        # Get the amplitudes and the indices of each peak
        peak_maxima_sign = np.sign(intensity[maxima_idx])
        peak_maxima_pos = peak_maxima_sign > 0
        peak_maxima_neg = peak_maxima_sign < 0
        
        if not peak_maxima_sign.dtype==float:
            raise TypeError(f"peak_maximas_sign must be float, got {peak_maxima_sign.dtype}")
        if not peak_maxima_pos.dtype==bool:
            raise TypeError(f"peak_maxima_pos must be bool, got {peak_maxima_pos.dtype}")
        if not peak_maxima_neg.dtype==bool:
            raise TypeError(f"peak_maxima_pos must be bool, got {peak_maxima_neg.dtype}")
        
        return peak_maxima_sign, peak_maxima_pos, peak_maxima_neg
        
 
    def validate_ranges(self, ranges:list[npt.NDArray[np.int64]], mask:npt.NDArray[np.bool_])->list[npt.NDArray[np.int64]]:
        
        validated_ranges = []
        for i, r in enumerate(ranges):
            if mask[i]:
                validated_ranges.append(r)
            
        return validated_ranges
    
    
    def window_signal_df(self,
                amp: npt.NDArray[np.float64],
                norm_amp: npt.NDArray[np.float64],
                time: npt.NDArray[np.float64],
                left:npt.NDArray[np.int64],
                right:npt.NDArray[np.int64],
                buffer:int=0,
                )->pt.DataFrame[BaseWindowedSignalDF]:

        amp = np.asarray(amp, dtype=np.float64)
        norm_amp = np.asarray(norm_amp, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        left = np.asarray(left,dtype=np.int64)
        right = np.asarray(right,dtype=np.int64)
        
        if len(amp)==0:
            raise ValueError("amplitude array has length 0")
        if len(norm_amp)==0:
            raise ValueError("norm_amp array has length 0")
        if len(time)==0:
            raise ValueError("time array has length 0")
        if len(left)==0:
            raise ValueError("left index array has length 0")
        if len(right)==0:
            raise ValueError("right index array has length 0")
        
        ranges = self.compute_peak_time_ranges(
            norm_amp,
            left,
            right,
            buffer
        )
        
        windowed_signal_df = pd.DataFrame(
            {
        "time_idx":pd.Series(np.arange(len(time)), dtype=pd.Int64Dtype()),
        "time":pd.Series(time, dtype=pd.Float64Dtype()),
        "amp":pd.Series(amp, dtype=pd.Float64Dtype()),
        "norm_amp": pd.Series(norm_amp, dtype=pd.Float64Dtype()),
        "window_id": pd.Series([0]*len(time), dtype = pd.Int64Dtype()), # default value is 0
        "window_type": pd.Series(['interpeak']*len(time),dtype=str), # default value is 'interpeak'
        "range_idx": pd.Series([], dtype=pd.Int64Dtype())
            }
            ).pipe(pt.DataFrame[BaseWindowedSignalDF])
        
        """
        every r in ranges is a subset of time_dx, therefore we can confidently
        expect to always return a value for a where call:
        
        windowed_signal_df['window_id'].where(windowed_signal_df['time_idx'].isin(r),i)
        
        window id corresponds to the region of the time domain wherein its assigned peak lies
        therefore the same id value stretches the same length as the identified peak, and zero for
        background regions. Example: 000111111110000022222222222000000333333300044444444440000, etc.
        
        Range order is the numerical ordering of the ranges from the start to the end of the signal
        
        window_id marks the location were a range is in the time index.
        
        So join on 'time_idx' and 'range_idx' to assign a range id to the time values
        """
        # turn ranges into a 2 column dataframe with a label column and range index. Label column should be in
        # blocks of values corresponding to each range being stacked on the other
        
        # so initially the frame is formed with each range as a row, producing a
        # jagged frame with lots of NA's. Use Pandas Nullable Integer type to handle the NA's.
        # We then transpose so each row is an observation and each column is a range.
        
        ranges_df = (pd.DataFrame(ranges, dtype=pd.Int64Dtype())
                     .T
                     .rename_axis(columns='window_id')
                     .rename_axis(index='i'))
        
        
        # now stack the frame so that the column labels become a second column and
        # the values are arranged vertically. Add a value of 1 to the 'window_id' to
        # reserve 0 for background regions.
        
        ranges_df = (ranges_df
                     .melt(value_name='range_idx')
                     .assign(**{'window_id':lambda df: df['window_id']+1})
                     )
        
        # we expect range_idx to be a subset of the time idx, throw an error if not

        range_idx_in_time_idx =ranges_df['range_idx'].isin(windowed_signal_df['time_idx'])
        
        if not range_idx_in_time_idx.any():
            raise ValueError("all values of 'range_idx' are expected to be in 'time_idx'\n"
                             "not in:"
                             f"{range_idx_in_time_idx}"
                             )
        
        # perform a left join on the time index and corresponding range index to label
        # those areas of the time domain belonging to an identified peak. Remaining areas
        # are NA after the join, so fill with 0 to label the background areas.
        
        try:
            windowed_signal_df = (pd.merge(
                left=windowed_signal_df.drop(['window_id','range_idx'],axis=1),
                right=ranges_df,
                left_on='time_idx',
                right_on='range_idx',
                how='left'
                )
                                .fillna(0) # background regions
                                .astype({'window_id':pd.Int64Dtype()})
                                .pipe(pt.DataFrame[BaseWindowedSignalDF]))
            
        except Exception as e:
            
            raise RuntimeError(e)

        # add string labels to windowed_signal_df - 'peak' if a peak region, 'interrpeak'
        # otherwise
        
        mask = ~(windowed_signal_df['window_id']==0)
        windowed_signal_df['window_type']=windowed_signal_df['window_type'].where(mask,'peak')
        
        # rearrange the columns so that 'window_type' is last
        cols = windowed_signal_df.columns.drop(['window_type']).tolist()
        cols.append('window_type')
        windowed_signal_df = windowed_signal_df.reindex(columns=cols).pipe(pt.DataFrame[BaseWindowedSignalDF])
        
        return windowed_signal_df.pipe(pt.DataFrame[BaseWindowedSignalDF])

    def compute_peak_time_ranges(self,
                                 norm_int: npt.NDArray[np.float64],
                                 left: npt.NDArray[np.float64],
                                 right: npt.NDArray[np.float64],
                                 buffer: int,
                                 )->list[npt.NDArray[np.int64]]:
        # calculate the range of each peak, returned as a list of ranges. Essentially
        # translates the calculated widths to time intervals, modified by the buffer
        
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
        validated_ranges = self.validate_ranges(ranges, ranges_mask)
        
        if len(validated_ranges)==0:
            raise ValueError("Something has gone wrong with the ranges or the validation")
                
        return validated_ranges
        