import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class WindowState:
    def __init__(self,
                 window_idx=None,
                 lb=None,
                 ub=None,
                 guess=None,
                 peak_window=None,
                 parorder=None,
                 full_windowed_chm_df=None,
                 ):
    
        self._window_idx=window_idx
        self._lb=lb
        self._ub=ub
        self._guess=guess
        
        self._parorder=parorder
        self._peak_window=peak_window
        self._time_range = self._peak_window['time_range']
        self._signal=self._peak_window['signal']
        self._signalarea = self._peak_window['signal_area']
        self._num_peaks = self._peak_window['num_peaks']
        self._amplitudes=self._peak_window['amplitude']
        self._locations=self._peak_window['location']
        self._widths = self._peak_window['width']
        
        self.full_windowed_chm_df = full_windowed_chm_df
        
        # self.assemble_peak_window_df()
        self.param_df = self.assemble_parameter_df()

        self.window_peak_properties_df = self.assemble_window_peak_properties_df()

        self.signal_df = pd.DataFrame(
            dict(
                tbl_name = 'signal',
                window_idx = self._window_idx,
                time=self._time_range,
                signal=self._signal
            )
            )
        
        self.window_info_df = pd.DataFrame(
            dict(
                tbl_name='window_info',
                window_idx=self._window_idx,
                num_peaks=self._num_peaks,
                signal_area=self._signalarea
            ),
            index=[0]
        )
        
        self._peak_fitting_info_df = pd.merge(left=self.window_peak_properties_df.drop(['tbl_name'],axis=1),
                                              right=self.param_df.drop(['tbl_name','window_idx'],axis=1),
                                              left_on='peak_idx',
                                              right_on='peak_idx',
                                              suffixes=['_l','_r'])
        
        self._peak_fitting_info_df.insert(0, 'tbl_name','window_peak_fit_info')
        
    def plot_full_windowed_signal(self):
        sns.relplot(
            self.full_windowed_chm_df
            .assign(window_id=lambda x: x.window_id.astype(str))
            .query("window_type=='peak'"),
            x='time',
            y='signal_corrected',
            hue='window_id',
            kind='line'    
        )
        plt.show()
        
        return None

    def assemble_parameter_df(self):
        
        # recreate peak index as a function of the length of the bounds array
        base_range = np.arange(self._num_peaks)+1
        
        peak_idx = np.repeat(base_range, len(self._parorder))
        
        parorder_idx = self._parorder*self._num_peaks
        
        # initialise the df
        
        parameter_df = pd.DataFrame(
            dict(
            tbl_name='parameters',
            window_idx = self._window_idx,
            peak_idx = peak_idx,
            params=parorder_idx,
            lb=self._lb,
            guess=self._guess,
            ub=self._ub,
            )
        )
        
        # add difference columns for lower and upper bound compared to guess. If guess
        # is less than lower bound, return a negative value, if guess is larger than
        # upper bound, return a negative value
        parameter_df = parameter_df.assign(
            delta_lb = lambda df: df.guess-df.lb,
            delta_ub = lambda df: df.ub-df.guess,
        )
        
        # assign a bool column where negative values are found
        
        condition = (parameter_df.delta_lb>0)&(parameter_df.delta_ub>0)
        
        parameter_df = parameter_df.assign(is_oob=0)
        parameter_df = parameter_df.assign(is_oob=lambda df: df.is_oob.where(
            condition,1))
        
        return parameter_df
        
    def plot_window(self):
        
        error_peak = self._peak_fitting_info_df.query('is_oob==1')
        width_start, width_end=error_peak.location-error_peak.width/2, error_peak.location+error_peak.width/2
        
        print(error_peak.to_markdown())
        
        fig, ax = plt.subplots(1)
        
        ax.plot(self.signal_df.time, self.signal_df.signal)
        ax.plot(self._peak_fitting_info_df.location, self._peak_fitting_info_df.amplitude, '.', label='peaks')
        
        ax.vlines(
            [width_start,width_end],
            ymin=error_peak.amplitude.max()*0.75,
            ymax=error_peak.amplitude.max()*1.25,
            label="width bounds of error peak",
            alpha=0.8,
            color='green',
            lw=0.75
                #    linestyles='dotted'
                   )
        
        ax.annotate("error peak", (error_peak.location+0.05, error_peak.amplitude))
        fig.suptitle(f"window {self._window_idx}")
        
        import matplotlib.ticker as plticker
        
        loc = plticker.MultipleLocator(0.25)
        ax.xaxis.set_major_locator(loc)
        ax.minorticks_on()
        ax.grid(axis='x', which='both')
        fig.legend()
        plt.show()
        
        return None
        
    def assemble_window_peak_properties_df(self):
        
        
        peak_window_df = pd.DataFrame(
         dict(
             tbl_name='peak_info',
             window_idx = self._window_idx,
             peak_idx = np.arange(1, len(self._amplitudes)+1),
             location=self._locations,
             amplitude=self._amplitudes,
             width=self._widths

         )
        )
        
        peak_window_df=peak_window_df.rename_axis('peak')
        return peak_window_df
        
    # def __str__(self):
    #     return f"{self.bounds}"
