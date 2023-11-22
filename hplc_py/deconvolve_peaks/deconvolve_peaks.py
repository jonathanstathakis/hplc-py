import numpy as np
import scipy
import warnings
import tqdm

from hplc_py.skewnorms import skewnorms

class PeakDeconvolver(skewnorms.SkewNorms):
    
    def add_custom_param_bounds(self, param_bounds, _param_bounds, parorder):
        for p in parorder:
            if p in param_bounds.keys():
                if p == 'amplitude':
                    _param_bounds[p] = v['amplitude'][i] * \
                        np.sort(param_bounds[p])
                elif p == 'location':
                    _param_bounds[p] = [v['location']
                                        [i] + p for p in param_bounds[p]]
                else:
                    _param_bounds[p] = param_bounds[p]
        
        return _param_bounds
    
    def add_peak_specific_bounds(self, i, known_peaks, parorder, paridx, v, p0, _param_bounds):
        # Add peak-specific bounds if provided
        if (type(known_peaks) == dict) & (len(known_peaks) != 0):
            if v['location'][i] in known_peaks.keys():
                newbounds = known_peaks[v['location'][i]]
                tweaked = False
                if len(newbounds) > 0:
                    for p in parorder:
                        if p in newbounds.keys():
                            _param_bounds[p] = np.sort(newbounds[p])
                            tweaked = True
                            if p != 'location':
                                p0[paridx[p]] = np.mean(newbounds[p])
            
                    # Check if width is the only key
                    if (len(newbounds) >= 1) & ('width' not in newbounds.keys()):
                        if tweaked == False:
                            raise ValueError(
                                f"Could not adjust bounds for peak at {v['location'][i]} because bound keys do not contain at least one of the following: `location`, `amplitude`, `scale`, `skew`. ")
                            
        return p0, _param_bounds
    
    def assemble_deconvolved_window_output(self, i, v, t_range, popt, window_dict):
        # Assemble the dictionary of output
        if v['num_peaks'] > 1:
            popt = np.reshape(popt, (v['num_peaks'], 4))
        else:
            popt = [popt]
        for i, p in enumerate(popt):
            window_dict[f'peak_{i + 1}'] = {
                'amplitude': p[0],
                'retention_time': p[1],
                'scale': p[2],
                'alpha': p[3],
                'area': self._compute_skewnorm(t_range, *p).sum(),
                'reconstructed_signal': self._compute_skewnorm(v['time_range'], *p)}
        
        return window_dict
    
    def get_many_peaks_warning(self):
        return warnings.warn(f"""
-------------------------- Hey! Yo! Heads up! ----------------------------------
| This time window (from {np.round(v['time_range'].min(), decimals=4)} to {np.round(v['time_range'].max(), decimals=3)}) has {v['num_peaks']} candidate peaks.
| This is a complex mixture and may take a long time to properly fit depending 
| on how well resolved the peaks are. Reduce `buffer` if the peaks in this      
| window should be separable by eye. Or maybe just go get something to drink.
--------------------------------------------------------------------------------
""")
    def build_initial_guess(self, p0, v, i):
        # Set up the initial guess
        p0.append(v['amplitude'][i])
        p0.append(v['location'][i]),
        p0.append(v['width'][i] / 2)  # scale parameter
        p0.append(0)  # Skew parameter, starts with assuming Gaussian
            
        return p0
    
    def build_default_bounds(self, i, v):
        _param_bounds = dict(
                        amplitude=np.sort([0.1 * v['amplitude'][i], 10 * v['amplitude'][i]]),
                        location=[v['time_range'].min(), v['time_range'].max()],
                        scale=[self._dt, (v['time_range'].max() - v['time_range'].min())/2],
                        skew=[-np.inf, np.inf])
        
        return _param_bounds

                
    def deconvolve_windows(self, k,v, param_bounds, known_peaks, parorder, paridx, max_iter, optimizer_kwargs, t_range):
        window_dict = {}

        p0 = []
        bounds = [[],  []]

        # If there are more than 5 peaks in a mixture, throw a warning
        if v['num_peaks'] >= 10:
            self.get_many_peaks_warning()
        
        for i in range(v['num_peaks']):
    
            p0 = self.build_initial_guess(p0, v, i)

            # Set default parameter bounds
            _param_bounds = self.build_default_bounds(i,v)
                
            # Modify the parameter bounds given arguments
            if len(param_bounds) != 0:
                _param_bounds = self.add_custom_param_bounds(param_bounds, _param_bounds, parorder)

            # modify the parameter bounds and guess based on peak specific user input
            p0, _param_bounds = self.add_peak_specific_bounds(i, known_peaks, parorder, paridx, v, p0, _param_bounds)
        
            # organise the bounds into arrays for scipy input
            for _, val in _param_bounds.items():
                bounds[0].append(val[0])
                bounds[1].append(val[1])
        
        # add _param_bounds to the class _param_bounds list
        self._param_bounds.append(_param_bounds)

        # Perform the inference
        popt, _ = scipy.optimize.curve_fit(self._fit_skewnorms, v['time_range'],
                                            v['signal'], p0=p0, bounds=bounds, maxfev=max_iter,
                                            **optimizer_kwargs)
    
        window_dict = self.assemble_deconvolved_window_output(i, v, t_range, popt, window_dict)
        
        return window_dict
    
    def find_integration_area(self, integration_window):
        # Determine the areas over which to integrate the window
        if len(integration_window) == 0:
            t_range = self.df[self.time_col].values
        elif (type(integration_window) == list):
            if len(integration_window) == 2:
                t_range = np.arange(
                    integration_window[0], integration_window[1], self._dt)
            else:
                raise RuntimeError(
                    'Provided integration bounds has wrong dimensions. Should have a length of 2.')
        return t_range    

    def deconvolve_peaks(self,
                         verbose=True,
                         known_peaks=[],
                         param_bounds={},
                         integration_window=[],
                         max_iter=1000000,
                         optimizer_kwargs={}):
        R"""
        .. note::
           In most cases, this function should not be called directly. Instead, 
           it should called through the :func:`~hplc_py.quant.Chromatogram.fit_peaks`

        For each peak window, estimate the parameters of skew-normal distributions 
        which makeup the peak(s) in the window. See "Notes" for information on
        default parameter bounds.

        Parameters
        ----------
        verbose : `bool`
            If `True`, a progress bar will be printed during the inference.

        param_bounds : `dict`
            Modifications to the default parameter bounds (see Notes below) as 
            a dictionary for each parameter. A dict entry should be of the 
            form `parameter: [lower, upper]`. Modifications have the following effects:
                * Modifications to `amplitude` bounds are multiplicative of the 
                  observed magnitude at the peak position. 
                * Modifications to `location` are values that are subtracted or 
                  added from the peak position for lower and upper bounds, respectively.
                * Modifications to `scale` replace the default values. 
                * Modifications to `skew` replace the default values. 
        integration_window: `list`
            The time window over which the integrated peak areas should be computed. 
            If empty, the area will be integrated over the entire duration of the 
            cropped chromatogram.
        max_iter : `int`
            The maximum number of iterations the optimization protocol should 
            take before erroring out. Default value is 10^6.

        optimizer_kwargs : `dict`
            Keyword arguments to be passed to `scipy.optimize.curve_fit`.

        Returns 
        --------
        peak_props: `dict`
            A dataframe containing properties of the peak fitting procedure. 

        Notes
        -----
        The parameter boundaries are set automatically to prevent run-away estimation 
        into non-realistic regimes that can seriously slow down the inference. The 
        default parameter boundaries for each peak are as follows.

            * `amplitude`: The lower and upper peak amplitude boundaries correspond to one-tenth and ten-times the value of the peak at the peak location in the chromatogram.

            * `location`: The lower and upper location bounds correspond to the minimum and maximum time values of the chromatogram.

            * `scale`: The lower and upper bounds of the peak standard deviation defaults to the chromatogram time-step and one-half of the chromatogram duration, respectively.  

            * `skew`: The skew parameter by default is allowed to take any value between (-`inf`, `inf`).
        """
        if self.window_props is None:
        
            raise RuntimeError(
                'Function `_assign_windows` must be run first. Go do that.')
        
        if verbose:
        
            self._fitting_progress_state = 1
            iterator = tqdm.tqdm(self.window_props.items(),
                                 desc='Deconvolving mixture')
        
        else:
            self._fitting_progress_state = 0
            iterator = self.window_props.items()
            
        parorder = ['amplitude', 'location', 'scale', 'skew']
        
        paridx = {k: -(i+1) for i, k in enumerate(reversed(parorder))}
        
        peak_props = {}
        
        self._bounds = []
        
        t_range = self.find_integration_area(integration_window=integration_window)

        # Instantiate a state variable to ensure that parameter bounds are being adjusted.
        self._param_bounds = []
        
        for k, v in iterator:
            window_dict = self.deconvolve_windows(k,v, param_bounds, known_peaks, parorder, paridx, max_iter, optimizer_kwargs, t_range)
            if v['num_peaks'] == 0:
                continue
            peak_props[k] = window_dict

        self._peak_props = peak_props
        
        return peak_props