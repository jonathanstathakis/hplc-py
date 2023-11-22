import scipy
import numpy as np
import warnings

class WindowFinder():
    def _assign_windows(self,
                        known_peaks=[],
                        tolerance=0.5,
                        prominence=0.01,
                        rel_height=1,
                        buffer=0,
                        peak_kwargs={}):
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
        if (rel_height < 0) | (rel_height > 1):
            raise ValueError(f' `rel_height` must be [0, 1].')

        # Correct for a negative baseline
        df = self.df
        intensity = self.df[self.int_col].values
        int_sign = np.sign(intensity)
        norm_int = (intensity - intensity.min()) / \
            (intensity.max() - intensity.min())
        self.normint = int_sign * norm_int

        # Preform automated peak detection and set window ranges
        peaks, _ = scipy.signal.find_peaks(
            int_sign * norm_int, prominence=prominence, **peak_kwargs)
        self._peak_indices = peaks

        # Get the amplitudes and the indices of each peak
        amps = np.sign(intensity[peaks])
        pos_inds = amps > 0
        neg_inds = amps < 0

        # Set up storage vectors for peak quantities
        _widths = np.zeros_like(amps)
        _left = np.zeros(len(amps)).astype(int)
        _right = np.zeros(len(amps)).astype(int)

        # Get the peak properties
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=scipy.signal._peak_finding_utils.PeakPropertyWarning)
            for i, inds in enumerate([pos_inds, neg_inds]):
                if i == 0:
                    _intensity = intensity
                else:
                    _intensity = -intensity
                if len(inds) > 0:
                    __widths, _, _, _ = scipy.signal.peak_widths(_intensity,
                                                                 self._peak_indices,
                                                                 rel_height=0.5)
                    _widths[inds] = __widths[inds]

                    _, _, __left, __right = scipy.signal.peak_widths(_intensity,
                                                                     self._peak_indices,
                                                                     rel_height=rel_height)
                    _left[inds] = __left[inds]
                    _right[inds] = __right[inds]

        # Determine if peaks should be added.
        if len(known_peaks) > 0:
            # Get the enforced peak positions
            if type(known_peaks) == dict:
                _known_peaks = list(known_peaks.keys())
            else:
                _known_peaks = known_peaks

            # Find the nearest location in the time array given the user-specified time
            enforced_location_inds = np.int_(
                np.array(_known_peaks) / self._dt) - self._crop_offset

            # Update the user specified times with the nearest location
            updated_loc = self._dt * enforced_location_inds + self._crop_offset
            if type(known_peaks) == dict:
                updated_known_peaks = known_peaks.copy()
                for _new, _old in zip(updated_loc, _known_peaks):
                    updated_known_peaks[_new] = updated_known_peaks.pop(_old)
                _known_peaks = list(updated_known_peaks)
            else:
                updated_known_peaks = list(np.copy(known_peaks))
                for _old, _new in enumerate(updated_loc):
                    updated_known_peaks[_old] = _new
                _known_peaks = updated_known_peaks
            self._known_peaks = updated_known_peaks

            # Clear peacks that are within a tolerance of the provided locations
            # of known peaks.
            for i, loc in enumerate(enforced_location_inds):
                autodetected = np.nonzero(
                    np.abs(self._peak_indices - loc) <= (tolerance / self._dt))[0]
                if len(autodetected) > 0:
                    # Remove the autodetected peak
                    self._peak_indices = np.delete(
                        self._peak_indices, autodetected[0])
                    _widths = np.delete(_widths, autodetected[0])
                    _left = np.delete(_left, autodetected[0])
                    _right = np.delete(_right, autodetected[0])

            # Add the provided locations of known peaks and adjust parameters as necessary.
            for i, loc in enumerate(enforced_location_inds):
                self._peak_indices = np.append(self._peak_indices, loc)
                if self._added_peaks is None:
                    self._added_peaks = []
                self._added_peaks.append((loc + self._crop_offset) * self._dt)
                if type(known_peaks) == dict:
                    _sel_loc = updated_known_peaks[_known_peaks[i]]
                    if 'width' in _sel_loc.keys():
                        _widths = np.append(
                            _widths, _sel_loc['width'] / self._dt)
                    else:
                        _widths = np.append(_widths, 1 / self._dt)
                else:
                    _widths = np.append(_widths, 1 / self._dt)
                _left = np.append(_left, loc - _widths[-1] - buffer)
                _right = np.append(_right, loc + _widths[-1] + buffer)
        else:
            self._known_peaks = known_peaks

        # Set window ranges
        ranges = []
        for l, r in zip(_left, _right):
            _range = np.arange(int(l - buffer), int(r + buffer), 1)
            _range = _range[(_range >= 0) & (_range <= len(norm_int))]
            ranges.append(_range)

        # Identiy subset ranges and remove
        valid = [True] * len(ranges)
        if len(ranges) > 1:
            for i, r1 in enumerate(ranges):
                for j, r2 in enumerate(ranges):
                    if i != j:
                        if set(r2).issubset(r1):
                            valid[j] = False

        # Keep only valid ranges and baselines
        ranges = [r for i, r in enumerate(ranges) if valid[i] is True]

        # Copy the dataframe and return the windows
        window_df = df.copy(deep=True)
        window_df.sort_values(by=self.time_col, inplace=True)
        window_df['time_idx'] = np.arange(len(window_df))
        window_df['window_id'] = 0
        window_df['window_type'] = 'peak'
        self.ranges = ranges
        for i, r in enumerate(ranges):
            window_df.loc[window_df['time_idx'].isin(r),
                          'window_id'] = int(i + 1)

        # Determine the windows for the background (nonpeak) areas.
        bg_windows = window_df[window_df['window_id'] == 0]
        tidx = bg_windows['time_idx'].values

        if len(bg_windows) > 0:
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
        window_df = window_df[window_df['window_id'] > 0]

        # Convert this to a dictionary for easy parsing
        window_dict = {}
        for g, d in window_df[window_df['window_type'] == 'peak'].groupby('window_id'):
            if g > 0:
                _peaks = [
                    p for p in self._peak_indices if p in d['time_idx'].values]
                peak_inds = [x for _p in _peaks for x in np.where(
                    self._peak_indices == _p)[0]]
                _dict = {'time_range': d[self.time_col].values,
                         'signal': d[self.int_col].values,
                         'signal_area': d[self.int_col].values.sum(),
                         'num_peaks': len(_peaks),
                         'amplitude': [d[d['time_idx'] == p][self.int_col].values[0] for p in _peaks],
                         'location': [d[d['time_idx'] == p][self.time_col].values[0] for p in _peaks],
                         'width':  [_widths[ind] * self._dt for ind in peak_inds]}
                window_dict[int(g)] = _dict

        self.window_df = window_df
        self.window_props = window_dict
        return window_df