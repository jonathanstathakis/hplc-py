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
from pandera.typing import Series

from dataclasses import dataclass, fields


from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl

from typing import Callable

import termcolor

@dataclass
class FitAssessment:
    def assess_fit(
        self,
        mixed_signal_df: DataFrame,
        unmixed_signal_df: DataFrame,
        window_df: DataFrame,
        rtol: float = 1e-2,
        ftol: float = 1e-2,
    ):
        score_df = self._score_df_factory(mixed_signal_df, unmixed_signal_df, window_df)

        score_df["applied_tolerance"] = rtol
        score_df.astype({"applied_tolerance": pd.Float64Dtype()})
        
        groups = ['window_type','window_idx']

        score_df = self.assign_tolpass(score_df, groups, rtol)
        score_df = self.assign_fanopass(score_df, groups, ftol)
        score_df = self.assign_status(score_df)
        

        fit_report = pd.DataFrame()
        
        self.print_report_card(score_df)
        return score_df

    def print_report_card(
        self,
        score_df: DataFrame,
    ):
        """
        Assemble a series of strings for printing a final report
        """

        grading = {
            "valid":"A+ Success: ",
            "invalid":"F, Failed: ",
            "needs review": "C-, Needs Review: ",
        }
        
        grading_colors = {
            "valid": ("black", "on_green"),
            "invalid": ("black", "on_red"),
            "needs review": ("black", "on_yellow"),
        }
        
        def review_warning(w_idx: int):
            return f"Interpeak window {w_idx} is not well reconstructed by mixture, but has a small Fano factor compared to peak region(s). This is likely acceptable, but visually check this region.\n"
        
        def invalid_warning(w_idx: int):
            return f"Interpeak window {w_idx} is not well reconstructed by mixture and has an appreciable Fano  factor compared to peak region(s). This suggests you have missed a peak in this region. Consider adding manual peak positioning by passing `known_peaks` to `fit_peaks()`"
        
        def valid_warning(w_idx: int):
            return ""
        
        warnings = {
            "valid":valid_warning,
            "invalid":invalid_warning,
            "needs review": review_warning,
            }
                
        def gen_report_str(x: Series,
                           grading: dict[str,str],
                           grading_colors: dict[str, tuple[str]],
                           warnings: dict[str, Callable]
                           ):
            
            columns = [
                'grading',
                'window',
                'time_range',
                'scores',
                'warning'
            ]
            
            rs = pd.DataFrame(index=x.index, columns=columns)
            
            rs['grading'] = grading[x.status.iloc[0]]
            rs['window'] = f"{x['window_type'].iloc[0]} window  {x['window_idx'].iloc[0]}"
            rs['time_range'] = f"(t: {x['time_start'].iloc[0]} - {x['time_end'].iloc[0]})"
            rs['scores'] = f"R-score = {x['reconstruction_score'].iloc[0]} & Fano Ratio = {x['signal_fano_factor'].iloc[0]}"
            rs['warning'] = warnings[x['status'].iloc[0]](x['window_idx'].iloc[0])
            
            rs_list = rs.iloc[0].astype(str).tolist()
            report_str = " ".join(rs_list)
            c_report_str = termcolor.colored(report_str, *grading_colors[x['status'].iloc[0]], attrs=['bold'])
            
            return c_report_str
        
        report_str = ""
        report_str += "-------------------Chromatogram Reconstruction Report Card----------------------\n\n"
        report_str += "Reconstruction of Peaks\n"
        report_str += "=======================\n\n"

        c_report_strs = score_df.groupby(['window_type','window_idx']).apply(
            gen_report_str,
            grading,
            grading_colors,
            warnings,
            )
        c_report_strs.name = "report_strs"
        
        # join the peak strings followed by a interpeak subheader then the interpeak strings
        c_report_strs = c_report_strs.reset_index()
        c_report_str = report_str + "\n".join(c_report_strs.loc[c_report_strs['window_type']=='peak', 'report_strs'])
        
        c_report_str += "\n\n"
        
        c_report_str += "Signal Reconstruction of Interpeak Windows\n"
        c_report_str += "=======================\n\n"
        
        c_report_str += "\n".join(c_report_strs.loc[c_report_strs['window_type']=='interpeak', 'report_strs'])
        
        
        import os
        os.system('color')
        termcolor.cprint(c_report_str, end='\n\n')
        
        return c_report_str
        
        
        
        

    def assign_status(
        self,
        score_df: DataFrame,
    ):
        """
        assign status of window based on tolpass and fanopass. if tolpass is invalid but fanopass is valid, assign 'needs review', else 'invalid'

        Treat the peaks and nonpeaks differently. Peaks are valid if tolpass is valid, nonpeaks are valid if tolpass valid, but need review
        """

        score_df["status"] = "invalid"

        valid_mask = score_df["tolpass"]

        # check fof tolpass validity
        score_df["status"] = score_df["status"].mask(valid_mask == True, "valid")
        
        review_mask =  (score_df["tolpass"]==False) | (score_df["fanopass"]==True)
        
        score_df["status"] = score_df["status"].mask(review_mask, "needs review")

        return score_df

    def assign_fanopass(
        self,
        score_df: DataFrame,
        groups: list[str],
        ftol: float,
    ):
        mean_fano_mask = score_df["window_type"] == "peak", "signal_fano_factor"
        mean_peak_fano = score_df.loc[mean_fano_mask].mean()

        score_df["fano_div"] = score_df.groupby(groups)["signal_fano_factor"].transform(
            lambda x: x.div(mean_peak_fano)
        )
        score_df["fanopass"] = score_df.groupby(groups)["fano_div"].transform(
            lambda x: x <= ftol
        )

        return score_df

    def assign_tolpass(
        self,
        score_df: DataFrame,
        groups: list[str],
        rtol: float,
    ):
        # determine whether the reconstruction score.. something?

        decimals = int(np.abs(np.ceil(np.log10(rtol))))

        score_df["rtol"] = rtol
        score_df["rounded_recon_score"] = score_df.groupby(groups)[
            "reconstruction_score"
        ].transform(lambda x: x.abs().sub(1).round(decimals))
        score_df["tolpass"] = (
            score_df.groupby(groups)
            .apply(lambda x: x["rounded_recon_score"] <= rtol)
            .to_numpy()
        )

        return score_df

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

    def prep_unmixed(self, unmixed_signal_df: DataFrame) -> DataFrame:
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
    ) -> DataFrame:
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

        ws_df = ws_df.set_index(["window_type", "window_idx", "time"]).reset_index()

        ws_df = ws_df.drop(["sw_idx"], axis=1)

        ws_df = ws_df.astype(
            {
                "time": pd.Float64Dtype(),
                "amp_mixed": pd.Float64Dtype(),
                "amp_unmixed": pd.Float64Dtype(),
            }
        )

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
    df = pd.DataFrame()

    groups = ["window_type", "window_idx"]
    vals = ["amp_mixed", "amp_unmixed"]

    df["time_start"] = signals.groupby(groups)["time"].min()

    df["time_end"] = signals.groupby(groups)["time"].max()

    df["signal_area"] = signals.groupby(groups)["amp_mixed"].apply(
        lambda x: np.abs(x.to_numpy()).sum() + 1
    )

    df["inferred_area"] = signals.groupby(groups)["amp_unmixed"].apply(
        lambda x: np.abs(x.to_numpy()).sum() + 1
    )

    df["signal_variance"] = signals.groupby(groups)["amp_mixed"].apply(
        lambda x: np.var(np.abs(x.to_numpy()))
    )

    df["signal_mean"] = signals.groupby(groups)["amp_mixed"].apply(
        lambda x: np.mean(np.abs(x.to_numpy()))
    )

    df["signal_fano_factor"] = df.groupby(groups).apply(
        lambda x: (x["signal_variance"].to_numpy() / x["signal_mean"].to_numpy())[0]
    )

    df["reconstruction_score"] = df.groupby(groups).apply(
        lambda x: (x["inferred_area"].to_numpy() / x["signal_area"].to_numpy())[0]
    )

    df = df.sort_index(ascending=[False, True]).reset_index()

    # convert floats to Floats
    df = df.astype(
        {col: pd.Float64Dtype() for col in df if pd.api.types.is_float_dtype(df[col])}
    )

    return df


def window_viz(
    df: DataFrame,
    groups: list | str,
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
