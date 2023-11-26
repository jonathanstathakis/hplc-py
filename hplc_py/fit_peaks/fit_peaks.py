import pandas as pd
import numpy as np
from hplc_py.find_windows import find_windows
from hplc_py.deconvolve_peaks import deconvolve_peaks
from hplc_py.baseline_correct import correct_baseline

class PeakFitter(correct_baseline.BaselineCorrector, find_windows.WindowFinder, deconvolve_peaks.PeakDeconvolver):

    def fit_peaks(self,
                    df: pd.DataFrame,
                    known_peaks=[],
                    tolerance=0.5,
                    prominence=1E-2,
                    rel_height=1,
                    approx_peak_width=5,
                    buffer=0,
                    param_bounds={},
                    integration_window=[],
                    verbose=True,
                    return_peaks=True,
                    correct_baseline=True,
                    max_iter=1000000,
                    precision=9,
                    peak_kwargs={},
                    optimizer_kwargs={}):
        R"""
        Detects and fits peaks present in the chromatogram

        Parameters
        ----------
        known_peaks : `list` or `dict`
            The approximate locations of peaks whose position is known. If 
            provided as a list, only the locations wil be used as initial guesses. 
            If provided as a dictionary, locations and parameter bounds will be 
            set.
        tolerance: `float`, optional
            If an enforced peak location is within tolerance of an automatically 
            identified peak, the automatically identified peak will be preferred. 
            This parameter is in units of time. Default is one-half time unit.
        prominence : `float`,  [0, 1]
            The promimence threshold for identifying peaks. Prominence is the 
            relative height of the normalized signal relative to the local
            background. Default is 1%. If `locations` is provided, this is 
            not used.
        rel_height : `float`, [0, 1]
            The relative height of the peak where the baseline is determined. This
            is used to split into windows and is *not* used for peak detection.
            Default is 100%. 
        approx_peak_width: `float`, optional
            The approximate width of the signal you want to quantify. This is 
            used as filtering window for automatic baseline correction. If `correct_baseline==False`,
            this has no effect. 
        buffer : positive `int`
            The padding of peak windows in units of number of time steps. Default 
            is 100 points on each side of the identified peak window. Must have a value 
            of at least 10.
        verbose : `bool`
            If True, a progress bar will be printed during the inference. 
        param_bounds: `dict`, optional
            Parameter boundary modifications to be used to constrain fitting of 
            all peaks. 
            See docstring of :func:`~hplc_py.quant.Chromatogram.deconvolve_peaks`
            for more information.
        integration_window: `list`
            The time window over which the integrated peak areas should be computed. 
            If empty, the area will be integrated over the entire duration of the 
            cropped chromatogram.
        correct_baseline : `bool`, optional
            If True, the baseline of the chromatogram will be automatically 
            corrected using the SNIP algorithm. See :func:`~hplc_py.quant.Chromatogram.correct_baseline`
            for more information.
        return_peaks : `bool`, optional
            If True, a dataframe containing the peaks will be returned. Default
            is True.
        max_iter : `int`
            The maximum number of iterations the optimization protocol should 
            take before erroring out. Default value is 10^6.
        precision : `int`
            The number of decimals to round the reconstructed signal to. Default
            is 9.
        peak_kwargs : `dict`
            Additional arguments to be passed to `scipy.signal.find_peaks`.
        optimizer_kwargs : `dict`
            Additional arguments to be passed to `scipy.optimize.curve_fit`.

        Returns
        -------
        peak_df : `pandas.core.frame.DataFrame`
            A dataframe containing information for each detected peak. This is
            only returned if `return_peaks == True`. The peaks are always 
            stored as an attribute `peak_df`.

        Notes
        -----
        This function infers the parameters defining skew-norma distributions 
        for each peak in the chromatogram. The fitted distribution has the form 

        .. math:: 
            I = 2S_\text{max} \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)e^{-\frac{(t - r_t)^2}{2\sigma^2}}\left[1 + \text{erf}\frac{\alpha(t - r_t)}{\sqrt{2\sigma^2}}\right]

        where :math:`S_\text{max}` is the maximum signal of the peak, 
        :math:`t` is the time, :math:`r_t` is the retention time, :math:`\sigma`
        is the scale parameter, and :math:`\alpha` is the skew parameter.

        """
        
        self._fit_peak_params = pd.DataFrame(
            dict(tbl_name='fit_peak_input_params',
                tolerance=tolerance,
                prominence=prominence,
                rel_height=rel_height,
                approx_peak_width=approx_peak_width,
                buffer=buffer,
                max_iter=max_iter,
                precision=precision
            ),
            index=[0]
            ).T
        
        if correct_baseline and not self._bg_corrected:
            self.correct_baseline(window=approx_peak_width,
                                    verbose=verbose, return_df=False)

        # Assign the window bounds
        _ = self._assign_windows(known_peaks=known_peaks,
                                    tolerance=tolerance,
                                    prominence=prominence, rel_height=rel_height,
                                    buffer=buffer, peak_kwargs=peak_kwargs)

        # Infer the distributions for the peaks
        peak_props = self.deconvolve_peaks(verbose=verbose,
                                            known_peaks=self._known_peaks,
                                            param_bounds=param_bounds,
                                            max_iter=max_iter,
                                            integration_window=integration_window,
                                            **optimizer_kwargs)

        # Set up a dataframe of the peak properties
        peak_df = pd.DataFrame([])
        iter = 0
        for _, peaks in peak_props.items():
            for _, params in peaks.items():
                _dict = {'retention_time': params['retention_time'],
                            'scale': params['scale'],
                            'skew': params['alpha'],
                            'amplitude': params['amplitude'],
                            'area': params['area']}
                iter += 1
                peak_df = pd.concat([peak_df, pd.DataFrame(_dict, index=[0])])

        peak_df.sort_values(by='retention_time', inplace=True)
        peak_df['peak_id'] = np.arange(len(peak_df)) + 1
        peak_df['peak_id'] = peak_df['peak_id'].astype(int)
        self.peaks = peak_df

        # Compute the mixture
        time = df[self.time_col].values
        out = np.zeros((len(time), len(peak_df)))
        iter = 0
        for _, _v in self._deconvolved_peak_props.items():
            for _, v in _v.items():
                params = [v['amplitude'], v['retention_time'],
                            v['scale'], v['alpha']]
                out[:, iter] = self._compute_skewnorm(time, *params)
                iter += 1
        
        # round the individual signals to the specified precision
        
        self.unmixed_chromatograms = np.round(out, decimals=precision)
        
        # assemble the individual signals as a frame with the peak ids and time axis.
        
        self.unmixed_chromatograms = pd.DataFrame(self.unmixed_chromatograms, columns=self.peaks.peak_id.values, index=time)
        
        
        # compute the maxima of each peak and add to the peak_df
        
        peak_df.insert(0, 'maxima', self.unmixed_chromatograms.max().values)
        
        self.peak_df = peak_df[['peak_id','retention_time','maxima','area','amplitude','scale','skew']].reset_index(drop=True)
        
        if return_peaks:
            return peak_df