import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import scipy
import warnings
import tqdm
from dataclasses import dataclass

from hplc_py.skewnorms import skewnorms
from hplc_py.deconvolve_peaks import windowstate

import logging
logger = logging.getLogger(__name__)
    
class PeakDeconvolver(skewnorms.SkewNorms):
    
    def add_custom_param_bounds(self, peak_idx, peak_window, param_bounds, _param_bounds, parorder):
        
        if not set(param_bounds.keys()).issubset(_param_bounds.keys()):
            raise ValueError(f"param_bounds must be one of {_param_bounds.keys()}, got {param_bounds.keys()}")
            
        
        for param in parorder:
            if param in param_bounds.keys():
                if param == 'amplitude':
                    _param_bounds[param] = peak_window['amplitude'][peak_idx] * \
                        np.sort(param_bounds[param])
                elif param == 'location':
                    _param_bounds[param] = [peak_window['location']
                                        [peak_idx] + p for p in param_bounds[param]]
                else:
                    _param_bounds[param] = param_bounds[param]
        
        return _param_bounds
    
    def add_peak_specific_bounds(self, peak_idx, known_peaks, parorder, paridx, peak_window, init_guess, _param_bounds):
        # Add peak-specific bounds if provided
        if (type(known_peaks) == dict) & (len(known_peaks) != 0):
            if peak_window['location'][peak_idx] in known_peaks.keys():
                newbounds = known_peaks[peak_window['location'][peak_idx]]
                tweaked = False
                if len(newbounds) > 0:
                    for p in parorder:
                        if p in newbounds.keys():
                            _param_bounds[p] = np.sort(newbounds[p])
                            tweaked = True
                            if p != 'location':
                                init_guess[paridx[p]] = np.mean(newbounds[p])
            
                    # Check if width is the only key
                    if (len(newbounds) >= 1) & ('width' not in newbounds.keys()):
                        if tweaked == False:
                            raise ValueError(
                                f"Could not adjust bounds for peak at {peak_window['location'][peak_idx]} because bound keys do not contain at least one of the following: `location`, `amplitude`, `scale`, `skew`. ")
                            
        return init_guess, _param_bounds
    
    def assemble_deconvolved_window_output(self, peak_idx, peak_window, t_range, popt, window_dict):
        # Assemble the dictionary of output
        if peak_window['num_peaks'] > 1:
            popt = np.reshape(popt, (peak_window['num_peaks'], 4))
        else:
            popt = [popt]
        for peak_idx, p in enumerate(popt):
            window_dict[f'peak_{peak_idx + 1}'] = {
                'amplitude': p[0],
                'retention_time': p[1],
                'scale': p[2],
                'alpha': p[3],
                'area': self._compute_skewnorm(t_range, *p).sum(),
                'reconstructed_signal': self._compute_skewnorm(peak_window['time_range'], *p)}
        
        return window_dict
    
    def get_many_peaks_warning(self, window_dict):
        return warnings.warn(f"""
-------------------------- Hey! Yo! Heads up! ----------------------------------
| This time window (from {np.round(window_dict['time_range'].min(), decimals=4)} to {np.round(window_dict['time_range'].max(), decimals=3)}) has {window_dict['num_peaks']} candidate peaks.
| This is a complex mixture and may take a long time to properly fit depending 
| on how well resolved the peaks are. Reduce `buffer` if the peaks in this      
| window should be separable by eye. Or maybe just go get something to drink.
--------------------------------------------------------------------------------
""")
    def build_initial_guess(self, init_guess, peak_window, peak_idx):
        # Set up the initial guess
        init_guess.append(peak_window['amplitude'][peak_idx])
        init_guess.append(peak_window['location'][peak_idx]),
        init_guess.append(peak_window['width'][peak_idx] / 2)  # scale parameter
        init_guess.append(0)  # Skew parameter, starts with assuming Gaussian
            
        return init_guess
    
    def build_default_bounds(self, peak_idx, peak_window):
        
        _peak_bounds = dict(
                        amplitude=np.sort([0.1 * peak_window['amplitude'][peak_idx], 10 * peak_window['amplitude'][peak_idx]]),
                        location=[peak_window['time_range'].min(), peak_window['time_range'].max()],
                        scale=[self._dt, (peak_window['time_range'].max() - peak_window['time_range'].min())/2],
                        skew=[-np.inf, np.inf])
        
        return _peak_bounds

                
    def deconvolve_windows(self, peak_window_id, window_dict, param_bounds, known_peaks, parorder, paridx, max_iter, optimizer_kwargs, t_range):
        
        deconvolved_window_dict = {}
    
        p0 = []
        bounds = [[],  []]

        # If there are more than 5 peaks in a mixture, throw a warning
        if window_dict['num_peaks'] >= 10:
            self.get_many_peaks_warning(window_dict)
        
        
        assert window_dict['num_peaks']>0, f"\n{window_dict}"
        
        # build guess and bounds for the window
        for peak_idx in range(window_dict['num_peaks']):
            
            p0 = self.build_initial_guess(p0, window_dict, peak_idx)

            # Set default parameter bounds
            _peak_bounds = self.build_default_bounds(peak_idx,window_dict)
            
            # Modify the parameter bounds given arguments
            if len(param_bounds) != 0:
                _peak_bounds = self.add_custom_param_bounds(peak_idx, window_dict, param_bounds, _peak_bounds, parorder)

            # modify the parameter bounds and guess based on peak specific user input
            p0, _peak_bounds = self.add_peak_specific_bounds(peak_idx, known_peaks, parorder, paridx, window_dict, p0, _peak_bounds)
        
            # organise the bounds into arrays for scipy input
            for _, val in _peak_bounds.items():
                bounds[0].append(val[0])
                bounds[1].append(val[1])
            
        # capture the state of the current window DEBUGGING
        self.windowstates.append(
                windowstate.WindowState(
                                window_idx=peak_window_id,
                                lb = bounds[0],
                                ub = bounds[1],
                                guess=p0,
                                peak_window=window_dict,
                                parorder=parorder,
                                full_windowed_chm_df=window_df
                                )
                                )
        
        assert '_peak_bounds' in locals()
        
        # add _peak_bounds to the class _param_bounds list
        self._param_bounds.append(_peak_bounds)

        try:
            # Perform the inference
            
            logger.info(f"fitting window {peak_window_id}")
            
            popt, _ = scipy.optimize.curve_fit(self._fit_skewnorms, window_dict['time_range'],
                                                window_dict['signal'], p0=p0, bounds=bounds, maxfev=max_iter,
                                                **optimizer_kwargs)
            
        except Exception as e:
            print(e)
            
            print("# Window Parameters")
            print(self.windowstates[-1].window_info_df.to_markdown())
            
            print("# Window Peak Fitting Parameters")
            print(self.windowstates[-1]._peak_fitting_info_df.to_markdown())
            
            self.windowstates[-1].plot_window()
            
            # self.windowstates[-1].plot_full_windowed_signal()
            raise ValueError
        
        deconvolved_window_dict = self.assemble_deconvolved_window_output(peak_idx, window_dict, t_range, popt, deconvolved_window_dict)
        
        return deconvolved_window_dict
    
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
            window_prop_dict_iterator = tqdm.tqdm(self.window_props.items(),
                                 desc='Deconvolving mixture')
        
        else:
            self._fitting_progress_state = 0
            window_prop_dict_iterator = self.window_props.items()
            
        parorder = ['amplitude', 'location', 'scale', 'skew']
        
        paridx = {k: -(i+1) for i, k in enumerate(reversed(parorder))}
        
        deconvolved_peak_props = {}
        
        self._bounds = []
        
        t_range = self.find_integration_area(integration_window=integration_window)

        # Instantiate a state variable to ensure that parameter bounds are being adjusted.
        self._param_bounds = []
        
        for peak_window_id, peak_window in window_prop_dict_iterator:
            deconvolved_window_dict = self.deconvolve_windows(
                peak_window_id=peak_window_id,
                window_dict=peak_window,
                param_bounds=param_bounds,
                known_peaks=known_peaks,
                parorder=parorder,
                paridx=paridx,
                max_iter=max_iter,
                optimizer_kwargs=optimizer_kwargs,
                t_range=t_range,
                )
            if peak_window['num_peaks'] == 0:
                continue
            deconvolved_peak_props[peak_window_id] = deconvolved_window_dict

        self._deconvolved_peak_props = deconvolved_peak_props
        
        return deconvolved_peak_props