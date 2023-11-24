import pandas as pd
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.special
import tqdm
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import termcolor

from hplc_py.fit_peaks import fit_peaks
from hplc_py.map_peaks import map_peaks

class Chromatogram(
    map_peaks.PeakMapper,
    fit_peaks.PeakFitterMixin,
                   ):
    """
    Base class for the processing and quantification of an HPLC chromatogram.

    Attributes
    ----------
    df : `pandas.core.frame.DataFrame`    
        A Pandas DataFrame containing the chromatogram, minimally with columns 
        of time and signal intensity. 
    window_props : `dict`
       A dictionary of each peak window, labeled as increasing integers in 
       linear order. Each key has its own dictionary with the following keys:
    peaks : `pandas.core.frame.DataFrame` 
        A Pandas DataFrame containing the inferred properties of each peak 
        including the retention time, scale, skew, amplitude, and total
        area under the peak across the entire chromatogram.
    unmixed_chromatograms : `numpy.ndarray`
        A matrix where each row corresponds to a time point and each column corresponds
        to the value of the probability density for each individual peak. This 
        is used primarily for plotting in the `show` method. 
    quantified_peaks : `pandas.core.frame.DataFrame`
        A Pandas DataFrame with peak areas converted to 
    scores : `pandas.core.frame.DataFrame`
        A Pandas DataFrame containing the reconstruction scores and Fano factor 
        ratios for each peak and interpeak region. This is generated only afer 
        `assess_fit()` is called.
    """

    def __init__(self,
                 file,
                 time_window=None,
                 cols={'time': 'time',
                       'signal': 'signal'}):
        """
        Instantiates a chromatogram object on which peak detection and quantification
        is performed.

        Parameters
        ----------
        file: `str` or pandas.core.frame.DataFrame`
            The path to the csv file of the chromatogram to analyze or 
            the pandas DataFrame of the chromatogram. If None, a pandas DataFrame 
            of the chromatogram must be passed.
        dataframe : `pandas.core.frame.DataFrame`
            a Pandas DataFrame of the chromatogram to analyze. If None, 
            a path to the csv file must be passed
        time_window: `list` [start, end], optional
            The retention time window of the chromatogram to consider for analysis.
            If None, the entire time range of the chromatogram will be considered.
        cols: `dict`, keys of 'time', and 'signal', optional
            A dictionary of the retention time and intensity measurements 
            of the chromatogram. Default is 
            `{'time':'time', 'signal':'signal'}`. 
       """

        # Peform type checks and throw exceptions where necessary.
        if (type(file) is not pd.core.frame.DataFrame):
            raise RuntimeError(
                f'Argument must be a Pandas DataFrame. Argument is of type {type(file)}')

        # Assign class variables
        self.time_col = cols['time']
        self.int_col = cols['signal']

        # Load the chromatogram and necessary components to self.
        dataframe = file.copy()
        self.df = dataframe

        # Define the average timestep in the chromatogram. This computes a mean
        # but values will typically be identical.
        self._dt = np.mean(np.diff(dataframe[self.time_col].values))

       # Blank out vars that are used elsewhere
        self.window_props = None
        self.scores = None
        self._peak_indices = None
        self.peaks = None
        self._guesses = None
        self._bg_corrected = False
        self._mapped_peaks = None
        self._added_peaks = None
        self.unmixed_chromatograms = None
        self._crop_offset = 0
        
        # to store the list of WindowState classes prior to deconvolve peaks DEBUGGING
        self.windowstates = []
        
        # Prune to time window
        if time_window is not None:
            self.crop(time_window)
        else:
            self.df = dataframe

    def __repr__(self):
        trange = f'(t: {self.df[self.time_col].values[0]} - {self.df[self.time_col].values[-1]})'
        rep = f"""Chromatogram:"""
        if self._crop_offset > 0:
            rep += f'\n\t\u2713 Cropped {trange}'
        if self._bg_corrected:
            rep += f'\n\t\u2713 Baseline Subtracted'
        if self.peaks is not None:
            rep += f"\n\t\u2713 {self.peaks.peak_id.max()} Peak(s) Detected"
            if self._added_peaks is not None:
                rep += f'\n\t\u2713 Enforced Peak Location(s)'
        if self._mapped_peaks is not None:
            rep += f'\n\t\u2713 Compound(s) Assigned'
        return rep

    def crop(self,
             time_window=None,
             return_df=False):
        R"""
        Restricts the time dimension of the DataFrame in place.

        Parameters
        ----------
        time_window : `list` [start, end], optional
            The retention time window of the chromatogram to consider for analysis.
            If None, the entire time range of the chromatogram will be considered.
        return_df : `bool`
            If `True`, the cropped DataFrame is 

        Returns
        -------
        cropped_df : pandas DataFrame
            If `return_df = True`, then the cropped dataframe is returned.
        """
        if self.peaks is not None:
            raise RuntimeError("""
You are trying to crop a chromatogram after it has been fit. Make sure that you 
do this before calling `fit_peaks()` or provide the argument `time_window` to the `fit_peaks()`.""")
        if len(time_window) != 2:
            raise ValueError(
                f'`time_window` must be of length 2 (corresponding to start and end points). Provided list is of length {len(time_window)}.')
        if time_window[0] >= time_window[1]:
            raise RuntimeError(
                f'First index in `time_window` must be ≤ second index.')

        # Apply the crop and return
        self.df = self.df[(self.df[self.time_col] >= time_window[0]) &
                          (self.df[self.time_col] <= time_window[1])]
        # set the offset
        self._crop_offset = int(time_window[0] / self._dt)
        if return_df:
            return self.df

    def _score_reconstruction(self):
        R"""
        Computes the reconstruction score on a per-window and total chromatogram
        basis. 

        Returns
        -------
        score_df : `pandas.core.frame.DataFrame`
            A DataFrame reporting the scoring statistic for each window as well 
            as for the entire chromatogram. A window value of `0` corresponds 
            to the chromatogram regions which don't have peaks. A window 
            value of `-1` corresponds to the chromatogram as a whole

        Notes
        -----
        The reconstruction score is defined as
        ..math:: 

            R = \frac{\text{area of inferred mixture in window} + 1}{\text{area of observed signal in window} + 1} = \frac{\frac{\sum\limits_{i\in t}^t \sum\limits_{j \in N_\text{peaks}}^{N_\text{peaks}}2A_j \text{SkewNormal}(\alpha_j, r_{t_j}, \sigma_j) + 1}{\sum\limits_{i \in t}^t S_i + 1}

        where :math:`t` is the total time of the region, :math:`A`is the inferred 
        peak amplitude, :math:`\alpha` is the inferred skew paramter, :math:`r_t` is
        the inferred peak retention time, :math:`\sigma` is the inferred scale 
        parameter and :math:`S_i` is the observed signal intensity at time point
        :math:`i`. Note that the signal and reconstruction is cast to be positive
        to compute the score.  

        """
        columns = ['window_id', 'time_start', 'time_end', 'signal_area',
                   'inferred_area', 'signal_variance', 'signal_mean', 'signal_fano_factor', 'reconstruction_score']
        score_df = pd.DataFrame([])
        # Compute the per-window reconstruction

        for g, d in self.window_df[self.window_df['window_type'] == 'peak'].groupby('window_id'):
            # Compute the non-peak windows separately.
            window_area = np.abs(d[self.int_col].values).sum() + 1
            window_peaks = self._deconvolved_peak_props[g]
            window_peak_area = np.array(
                [np.abs(v['reconstructed_signal']) for v in window_peaks.values()]).sum() + 1
            score = np.array(window_peak_area / window_area).astype(float)
            signal_var = np.var(np.abs(d[self.int_col].values))
            signal_mean = np.mean(np.abs(d[self.int_col].values))
            # Account for an edge case to avoid dividing by zero
            if signal_mean == 0:
                signal_mean += 1E-9
            signal_fano = signal_var / signal_mean
            x = [g, d[self.time_col].min(),
                 d[self.time_col].max(), window_area, window_peak_area,
                 signal_var, signal_mean, signal_fano, score]
            _df = pd.DataFrame(
                {_c: _x for _c, _x in zip(columns, x)}, index=[g - 1])
            _df['window_type'] = 'peak'
            score_df = pd.concat([score_df, _df])

        # Compute the score for the non-peak regions
        nonpeak = self.window_df[self.window_df['window_type'] == 'interpeak']
        if len(nonpeak) > 0:
            for g, d in nonpeak.groupby('window_id'):
                total_area = np.abs(d[self.int_col].values).sum() + 1
                recon_area = np.sum(np.abs(self.unmixed_chromatograms), axis=1)[
                    d['time_idx'].values].sum() + 1
                nonpeak_score = recon_area / total_area
                signal_var = np.var(np.abs(d[self.int_col].values))
                signal_mean = np.mean(np.abs(d[self.int_col].values))
                # Account for an edge case to avoide dividing by zero
                if signal_mean == 0:
                    signal_mean += 1E-9
                signal_fano = signal_var / signal_mean

                # Add to score dataframe
                x = [g, d[self.time_col].min(),
                     d[self.time_col].max(),
                     total_area, recon_area, signal_var, signal_mean, signal_fano, nonpeak_score]
                _df = pd.DataFrame(
                    {c: xi for c, xi in zip(columns, x)}, index=[g - 1])
                _df['window_type'] = 'interpeak'
                score_df = pd.concat([score_df, _df])
        score_df['signal_area'] = score_df['signal_area'].astype(float)
        score_df['inferred_area'] = score_df['inferred_area'].astype(float)
        self.scores = score_df
        return score_df

    def assess_fit(self,
                   rtol=1E-2,
                   fano_tol=1E-2,
                   verbose=True):
        R"""
        Assesses whether the computed reconstruction score is adequate, given a tolerance.

        Parameters
        ----------
        rtol : `float`
            The tolerance for a reconstruction to be valid. This is the tolerated 
            deviation from a score of 1 which indicates a perfectly reconstructed
            chromatogram. 
        fano_tol : `float`
            The tolerance away from zero for evaluating the Fano factor of 
            inerpeak windows. See note below.
        verbose : `bool`
            If True, a summary of the fit will be printed to screen indicating 
            problematic regions if detected.

        Returns
        -------
        score_df : `pandas.core.frame.DataFrame`  
            A DataFrame reporting the scoring statistic for each window as well 
            as for the entire chromatogram. A window value of `0` corresponds 
            to the entire chromatogram. A column `accepted` with a boolean 
            value represents whether the reconstruction is within tolerance (`True`)
            or (`False`).

        Notes
        -----
        The reconstruction score is defined as

        .. math:: 
            R = \frac{\text{area of inferred mixture in window} + 1}{\text{area of observed signal in window} + 1} 

        where :math:`t` is the total time of the region, :math:`A` is the inferred 
        peak amplitude, :math:`\alpha` is the inferred skew paramter, :math:`r_t` is
        the inferred peak retention time, :math:`\sigma` is the inferred scale 
        parameter and :math:`S_i` is the observed signal intensity at time point
        :math:`i`. Note that the signal and reconstruction is cast to be positive
        to compute the score.  

        A reconstruction score of :math:`R = 1` indicates a perfect 
        reconstruction of the chromatogram. For practical purposes, a chromatogram
        is deemed to be adequately reconstructed if :math:`R` is within a tolerance
        :math:`\epsilon` of 1 such that

        .. math::
            \left| R - 1 \right| \leq \epsilon \Rightarrow \text{Valid Reconstruction}

        Interpeak regions may have a poor reconstruction score due to noise or
        short durations. To determine if this poor reconstruction score is due 
        to a missed peak, the signal Fano factor of the region is computed as 

        .. math::
            F = \frac{\sigma^2_{S}}{\langle S \rangle}.

        This is compared with the average Fano factor of :math:`N` peak windows such 
        that the Fano factor ratio is 

        .. math::
            \frac{F}{\langle F_{peak} \rangle} = \frac{\sigma^2_{S} / \langle S \rangle}{\frac{1}{N} \sum\limits_{i}^N \frac{\sigma_{S,i}^2}{\langle S_i \rangle}}.

        If the Fano factor ratio is below a tolerance `fano_tol`, then that 
        window is deemed to be noisy and peak-free. 
        """

        if self.unmixed_chromatograms is None:
            raise RuntimeError(
                "No reconstruction found! `.fit_peaks()` must be called first. Go do that.")

        # Compute the reconstruction score
        _score_df = self._score_reconstruction()

        # Apply the tolerance parameter
        _score_df['applied_tolerance'] = rtol
        score_df = pd.DataFrame([])
        mean_fano = _score_df[_score_df['window_type']
                              == 'peak']['signal_fano_factor'].mean()
        for g, d in _score_df.groupby(['window_type', 'window_id']):

            tolpass = np.round(np.abs(d['reconstruction_score'].values[0] - 1),
                               decimals=int(np.abs(np.ceil(np.log10(rtol))))) <= rtol

            d = d.copy()
            if g[0] == 'peak':
                if tolpass:
                    d['status'] = 'valid'
                else:
                    d['status'] = 'invalid'

            else:
                fanopass = (d['signal_fano_factor'].values[0] /
                            mean_fano) <= fano_tol
                if tolpass:
                    d['status'] = 'valid'
                elif fanopass:
                    d['status'] = 'needs review'
                else:
                    d['status'] = 'invalid'
            score_df = pd.concat([score_df, d], sort=False)

        # Define colors printing parameters to avoid retyping everything.
        print_colors = {'valid': ('A+, Success: ', ('black', 'on_green')),
                        'invalid': ('F, Failed: ', ('black', 'on_red')),
                        'needs review': ('C-, Needs Review: ', ('black', 'on_yellow'))}
        if verbose:
            self._report_card_progress_state = 1
            print("""
-------------------Chromatogram Reconstruction Report Card----------------------

Reconstruction of Peaks
======================= 
""")
        else:
            self._report_card_progress_state = 0

        for g, d in score_df[score_df['window_type'] == 'peak'].groupby('window_id'):
            status = d['status'].values[0]
            if status == 'valid':
                warning = ''
            else:
                warning = """
Peak mixture poorly reconstructs signal. You many need to adjust parameter bounds 
or add manual peak positions (if you have a shouldered pair, for example). If 
you have a very noisy signal, you may need to increase the reconstruction 
tolerance `rtol`."""
            if (d['reconstruction_score'].values[0] >= 10) | \
                    (d['reconstruction_score'].values[0] <= 0.1):
                if d['reconstruction_score'].values[0] == 0:
                    r_score = f'0'
                else:
                    r_score = f"10^{int(np.log10(d['reconstruction_score'].values[0]))}"
            else:
                r_score = f"{d['reconstruction_score'].values[0]:0.4f}"
            if verbose:
                termcolor.cprint(f"{print_colors[status][0]} Peak Window {int(g)} (t: {d['time_start'].values[0]:0.3f} - {d['time_end'].values[0]:0.3f}) R-Score = {r_score}",
                                 *print_colors[status][1], attrs=['bold'], end='')
                print(warning)

        if len(score_df[score_df['window_type'] == 'interpeak']) > 0:
            if verbose:
                print("""
Signal Reconstruction of Interpeak Windows
==========================================
                  """)
            for g, d in score_df[score_df['window_type'] == 'interpeak'].groupby('window_id'):
                status = d['status'].values[0]
                if status == 'valid':
                    warning = ''
                elif status == 'needs review':
                    warning = f"""
Interpeak window {g} is not well reconstructed by mixture, but has a small Fano factor  
compared to peak region(s). This is likely acceptable, but visually check this region.\n"""
                elif status == 'invalid':
                    warning = f"""
Interpeak window {g} is not well reconstructed by mixture and has an appreciable Fano 
factor compared to peak region(s). This suggests you have missed a peak in this 
region. Consider adding manual peak positioning by passing `known_peaks` 
to `fit_peaks()`."""

                if (d['reconstruction_score'].values[0] >= 10) | \
                        (d['reconstruction_score'].values[0] <= 0.1):
                    if d['reconstruction_score'].values[0] == 0:
                        r_score = f'0'
                    else:
                        r_score = f"10^{int(np.log10(d['reconstruction_score'].values[0]))}"
                else:
                    r_score = f"{d['reconstruction_score'].values[0]:0.4f}"
                if ((d['signal_fano_factor'].values[0] / mean_fano) > 10) | \
                   ((d['signal_fano_factor'].values[0] / mean_fano) <= 1E-5):
                    if (d['signal_fano_factor'].values[0] / mean_fano) == 0:
                        fano = '0'
                    else:
                        fano = f"10^{int(np.log10(d['signal_fano_factor'].values[0] / mean_fano))}"
                else:
                    fano = f"{d['signal_fano_factor'].values[0] / mean_fano:0.4f}"

                if verbose:
                    termcolor.cprint(f"{print_colors[status][0]} Interpeak Window {int(g)} (t: {d['time_start'].values[0]:0.3f} - {d['time_end'].values[0]:0.3f}) R-Score = {r_score} & Fano Ratio = {fano}",
                                     *print_colors[status][1], attrs=['bold'], end='')
                    print(warning)
        if verbose:
            print("""
--------------------------------------------------------------------------------""")
        return score_df

    def show(self,
             fig,
             ax,
             time_range=[],
             ):
        """
        Displays the chromatogram with mapped peaks if available.

        Parameters
        ----------
        time_range : `List`
            Adjust the limits to show a restricted time range. Should 
            be provided as two floats in the range of [`lower`, `upper`]. Note
            that this does not affect the chromatogram directly as in `crop`. 


        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The matplotlib figure object.
        ax : `matplotlib.axes._axes.Axes`
            The matplotlib axis object.
        """
        sns.set()

        # Set up the figure
        
        if not ax:
        
            fig, ax = plt.subplots(1, 1)
        
        ax.set_xlabel(self.time_col)
        if self._bg_corrected:
            self._viz_ylabel_subtraction_indication = True
            ylabel = f"{self.int_col.split('_corrected')[0]} (baseline corrected)"
        else:
            self._viz_ylabel_subtraction_indication = False
            ylabel = self.int_col
        ax.set_ylabel(ylabel)

        # Plot the raw chromatogram
        ax.plot(self.df[self.time_col], self.df[self.int_col], 'k-',
                label='raw chromatogram')
        
        # Compute the skewnorm mix
        self._viz_min_one_concentration = None
        if self.peaks is not None:
            self._viz_peak_reconstruction = True
            time = self.df[self.time_col].values
            
            # Plot the mix
            convolved = np.sum(self.unmixed_chromatograms, axis=1)
            
            ax.plot(time, convolved, 'r--', label='inferred mixture')
            
            for g, d in self.peaks.groupby('peak_id'):
                label = f'peak {int(g)}'
                if self._mapped_peaks is not None:
                    self._viz_mapped_peaks = True
                    if g in self._mapped_peaks.keys():
                        d = self.quantified_peaks[self.quantified_peaks['compound']
                                                  == self._mapped_peaks[g]]
                        if 'concentration' in d.keys():
                            self._viz_min_one_concentration = True
                            label = f"{self._mapped_peaks[g]}\n[{d.concentration.values[0]:0.3g}"
                        else:
                            if self._viz_min_one_concentration is None:
                                self._viz_min_one_concentration = False
                            label = f"{self._mapped_peaks[g]}"
                        if 'unit' in d.keys():
                            self._viz_unit_display = True
                            label += f" {d['unit'].values[0]}]"
                        else:
                            self._viz_unit_display = False
                            label += ']'
                    else:
                        self._viz_mapped_peaks = False
                        label = f'peak {int(g)}'
                else:
                    self._viz_mapped_peaks = False
                    
                ax.fill_between(time, self.unmixed_chromatograms.values[:, int(g) - 1], label=label,
                                alpha=0.5)
                
                
                ax.annotate(text=g, xy=[d.retention_time, d.maxima+2], fontsize=6)
                ax.set_ylim(bottom=-d.maxima.max()*0.2, top=d.maxima.max()*1.1)
        else:
            self._viz_peak_reconstruction = False
        if 'estimated_background' in self.df.keys():
            self._viz_subtracted_baseline = True
            ax.plot(self.df[self.time_col], self.df['estimated_background'],
                    color='dodgerblue', label='estimated background', zorder=1)
        else:
            self._viz_subtracted_baseline = False

        if self._added_peaks is not None:
            ymax = ax.get_ylim()[1]
            for i, loc in enumerate(self._added_peaks):
                if i == 0:
                    label = 'suggested peak location'
                else:
                    label = '__nolegend__'
                ax.vlines(loc, 0, ymax, linestyle='--',
                          color='dodgerblue', label=label)
                
        # ax.legend(bbox_to_anchor=(1.5, 1))
        # fig.patch.set_facecolor((0, 0, 0, 0))
        
        if len(time_range) == 2:
            self._viz_adjusted_xlim = True
            ax.set_xlim(time_range)
            # Determine the max min and max value of the chromatogram within range.
            _y = self.df[(self.df[self.time_col] >= time_range[0]) & (
                self.df[self.time_col] <= time_range[1])][self.int_col].values
            ax.set_ylim([ax.get_ylim()[0], 1.1 * _y.max()])
        
        
        else:
            self._viz_adjusted_xlim = False
        return [fig, ax]
