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
        self.widths = self._peak_window['width']
        
        self.full_windowed_chm_df = full_windowed_chm_df
        
        # self.assemble_peak_window_df()
        self.param_df = self.assemble_parameter_df()
        self.peak_window_df = self.assemble_peak_window_df()
        
    def plot_window(self):
        
        plt.plot(self._time_range, self._signal)
        plt.show()
        
        return None
        
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
        param_length = len(self._ub)/self._num_peaks
        base_range = np.arange(1, param_length+1, dtype=int)
        peak_idx = np.repeat(base_range, self._num_peaks)
        parorder_idx = self._parorder*self._num_peaks
        
        for x in [peak_idx, parorder_idx, self._lb, self._ub]:
            print(len(x))
        
        # initialise the df
        
        parameter_df = pd.DataFrame(
            dict(
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
        
    def assemble_peak_window_df(self):
        
        
        peak_window_df = pd.DataFrame(
            [pd.Series(val, name=key) for key, val in self._peak_window.items()]
        )
        
        return peak_window_df
        
    # def __str__(self):
    #     return f"{self.bounds}"
