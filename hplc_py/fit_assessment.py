"""
Flow:

1. assess_fit
2. compute reconstruction score
3. apply tolerance
4. calculate mean fano
6. Determine whether window reconstruction score and fano ratio is acceptable
7. create 'report card'
8. 

Notes:
- reconstruction score defined as unmixed_signal_AUC+1 divided by mixed_signal_AUC+1. 

"""

import pandas as pd

import numpy as np
import numpy.typing as npt

import pandera as pa
import pandera.typing as pt
from pandera.typing import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import OutPeakReportBase

from dataclasses import dataclass


from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl


@dataclass
class FitAssessment:
    def assess_fit(
        self,
        mixed_signal_df,
        unmixed_signal_df,
        peak_report_df,
        window_df,
    ):
        pass

        score_df = self._score_df_factory(
            mixed_signal_df, unmixed_signal_df, peak_report_df
        )

    def _score_df_factory(
        self,
        mixed_signal_df: DataFrame,
        unmixed_signal_df: DataFrame,
        window_df: DataFrame,
    ):
        ws_df = self.prep_ws_df(mixed_signal_df, unmixed_signal_df, window_df)

        # window_viz(ws_df, ['window_type','window_idx'], 'time','amp_mixed','amp_unmixed')

        # perform the aggregations
        
        score_df = calc_wdw_aggs(ws_df)
        # score
        
        return score_df
    
    def prep_unmixed(
        self,
        unmixed_signal_df: DataFrame)->DataFrame:
        # form a windowed table containing the mixed and unmixed signals. They will all be joined on the time axis

        # prepare teh unmixed dataframe
        unmixed = unmixed_signal_df.set_index(["peak_idx", "time_idx", "time"])
        unmixed = unmixed.unstack("peak_idx")
        unmixed = unmixed.sum(axis=1)
        unmixed = unmixed.reset_index(name="unmixed_amp")
        
        return unmixed
    
    def prep_mixed(
        self,
        mixed_signal_df: DataFrame,
                   ):
        # prepare the mixed dataframe
                mixed = mixed_signal_df.set_index(["time_idx", "time"])
                mixed = mixed[["amp_corrected"]]
                return mixed

    def prep_ws_df(
        self,
        mixed: DataFrame,
        unmixed: DataFrame,
        window_df: DataFrame,
    )-> DataFrame:
        unmixed = self.prep_unmixed(unmixed)
        mixed = self.prep_mixed(mixed)
        # join mixed signal_df to unmixed_signal df

        signals = mixed.join(unmixed.set_index(["time_idx", "time"]))
        signals = signals.reset_index().set_index(["time_idx"])
        
        ws_df = signals.join(
            window_df.set_index(["time_idx"]),
            how="left",
            validate="1:1",
        )

        ws_df = ws_df.rename(
            {
                "unmixed_amp": "amp_unmixed",
                "amp_corrected": "amp_mixed",
            },
            axis=1,
        )
        
        ws_df = ws_df.set_index(['window_type','window_idx','time']).reset_index()
        
        ws_df = ws_df.drop(['sw_idx'],axis=1)
        
        ws_df = ws_df.astype({
            "time":pd.Float64Dtype(),
            "amp_mixed":pd.Float64Dtype(),
            "amp_unmixed":pd.Float64Dtype(),
        })
        
        return ws_df

def plot_windows(
    df,
    x_col: str,
    height: float,
    colormap,
    ax: plt.Axes,
):
    anchor_x = df[x_col].min()
    anchor_y = 0
    width = df[x_col].max() - df[x_col].min()

    color = colormap.colors[int(df["window_idx"].iloc[0]) - 1]

    rt = Rectangle(
        xy=(anchor_x, anchor_y),
        width=width,
        height=height,
        color=color,
    )

    ax.add_patch(rt)

    return None


def plot_windowing(
    df: pt.DataFrame,
    group_labels: list | str,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
):
    """
    TODO:
    - [ ] add logic to differentiate interpeak and peak, currently using the same colors in the same order, not useful viz.
    """
    # Window Height
    
    
    wh = df[y_col].max()

    colormap = mpl.colormaps["Set2"].resampled(df.groupby(group_labels).ngroups)
    ax.plot(df[x_col], df[y_col], label="amp_unmixed")

    df.groupby(group_labels).apply(plot_windows, x_col, wh, colormap, ax)

    df.groupby(group_labels).apply(label_windows, x_col, df[y_col].max() * 0.75, ax)
    return None


def label_windows(
    df: pt.DataFrame,
    x_col: str,
    label_height: float,
    ax: plt.Axes,
):
    # label each window at its center and ~75% height
    label_x = df[x_col].mean()
    label_y = label_height
    ax.annotate(df["window_idx"].iloc[0], (label_x, label_y))

def get_window_times(
        window_df: DataFrame,
        signal_df: DataFrame,
    ):
        # use d for named return of aggregate functions
        def time_start(x):
            return x.min()

        def time_end(x):
            return x.max()

        wdw_times = window_df.set_index("time_idx")
        wdw_times = wdw_times.join(
            signal_df.set_index("time_idx").loc[:, ["time"]],
            how="left",
            validate="1:1",
        )
        wdw_times = wdw_times.reset_index()
        wdw_times = wdw_times.pivot_table(
            index="sw_idx", values="time", aggfunc=[time_start, time_end]
        ).droplevel(level=1, axis=1)
        wdw_times = wdw_times.reset_index()

        return wdw_times
    
def calc_wdw_aggs(
        signals: DataFrame,
    ):
        
        # Index(['window_id', 'time_start', 'time_end', 'signal_area', 'inferred_area', 'signal_variance', 'signal_mean', 'signal_fano_factor','reconstruction_score', 'window_type']
        
        # colname, func
        
        df = pd.DataFrame()
        
        groups = ["window_type", "window_idx"]
        vals = ["amp_mixed", "amp_unmixed"]

        df['time_start'] = signals.groupby(groups)['time'].min()
        
        df['time_end'] = signals.groupby(groups)['time'].max()
        
        df['signal_area'] = signals.groupby(groups)['amp_mixed'].apply(lambda x: np.abs(x.to_numpy()).sum()+1)
        
        df['inferred_area'] = signals.groupby(groups)['amp_unmixed'].apply(lambda x: np.abs(x.to_numpy()).sum()+1)
        
        df['signal_variance'] = signals.groupby(groups)['amp_mixed'].apply(lambda x: np.var(np.abs(x.to_numpy())))
        
        df['signal_mean'] = signals.groupby(groups)['amp_mixed'].apply(lambda x: np.mean(np.abs(x.to_numpy())))
        
        df['signal_fano_factor'] = df.groupby(groups).apply(lambda x: (x['signal_variance'].to_numpy()/x['signal_mean'].to_numpy())[0])
        
        
        df['reconstruction_score'] = df.groupby(groups).apply(lambda x: (x['inferred_area'].to_numpy()/x['signal_area'].to_numpy())[0])
        
        df = df.sort_index(ascending=[False, True]).reset_index()
        
        # convert floats to Floats
        df = df.astype({col: pd.Float64Dtype() for col in df if pd.api.types.is_float_dtype(df[col])})
        
        return df
    
def window_viz(
    df: DataFrame,
    groups: list|str,
    x: str,
    signal_1: str,
    signal_2: str,
):
        
    # plot the windows

    # plot the signal

    # for each window add a Rectangle patch.

    fig, axs = plt.subplots(2)

    plot_windowing(
        df,
        groups,
        x,
        signal_1,
        axs[0],
        )

    plot_windowing(
        df,
        groups,
        x,
        signal_2,
        axs[1],
        )

    plt.legend()
    plt.show()
    
    return None
