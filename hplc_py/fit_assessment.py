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

from hplc_py.io_validation import IOValid
from .deconvolve_peaks.schemas import PSignals

from hplc_py.hplc_py_typing.hplc_py_typing import (
    FitAssessScores,
)

from hplc_py.map_windows.schemas import X_Windowed

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
import polars as pl

import termcolor


class FitAssessment(IOValid):
    def __init__(self):
        self.grading = pl.DataFrame(
            {
                "status": ["valid", "invalid", "needs review"],
                "grade": ["A+, success", "F, failed", "C-, needs review"],
            }
        )
        self.grading_colors = pl.DataFrame(
            {
                "status": ["valid", "invalid", "needs review"],
                "color_tuple": [
                    "black, on_green",
                    "black, on_red",
                    "black, on_yellow",
                ],
            }
        )

        self.ws_sch = X_Windowed
        self.sc_sch = FitAssessScores

    @pa.check_types
    def assess_fit(
        self,
        ws: DataFrame[X_Windowed],
        rtol: float = 1e-2,
        ftol: float = 1e-2,
    ):
        score_df = self.calc_wdw_aggs(ws, rtol, ftol)

        # self.print_report_card(score_df)
        return score_df

    @pa.check_types
    def calc_wdw_aggs(
        self,
        ws: DataFrame[X_Windowed],
        rtol: float,
        ftol: float,
    ) -> DataFrame[FitAssessScores]:
        self._check_scalar_is_type(rtol, float)
        self._check_scalar_is_type(rtol, float)

        import polars as pl

        ws_: pl.DataFrame = pl.from_pandas(ws)

        t_idx: str = str(self.ws_sch.t_idx)

        if "amp_corrected" in ws.columns:
            mxd: str = str(self.ws_sch.amp_corrected)
        else:
            mxd: str = str(self.ws_sch.amp)

        # declare the column labels from the schemas to improve readability of following
        # method chain. Allows final names to be declared in the Schema rather than locally.
        t: str = str(self.ws_sch.time)
        unmx: str = str(self.ws_sch.amp)
        w_type: str = str(self.ws_sch.w_type)
        w_idx: str = str(self.ws_sch.w_idx)

        mx_var: str = str(self.sc_sch.mixed_var)
        ts: str = str(self.sc_sch.time_start)
        ts: str = str(self.sc_sch.time_start)
        te: str = str(self.sc_sch.time_end)
        mx_u: str = str(self.sc_sch.mixed_mean)
        unmx_a: str = str(self.sc_sch.inferred_area)
        mx_a: str = str(self.sc_sch.signal_area)
        rc_sc: str = str(self.sc_sch.recon_score)
        rtol_l: str = str(self.sc_sch.rtol)
        tolchk: str = str(self.sc_sch.tolcheck)
        mx_fano: str = str(self.sc_sch.mixed_fano)
        u_fano: str = str(self.sc_sch.u_peak_fano)
        div_fano: str = str(self.sc_sch.fano_div)
        div_fano: str = str(self.sc_sch.fano_div)
        tolpass: str = str(self.sc_sch.tolpass)
        fanopass: str = str(self.sc_sch.fanopass)
        status: str = str(self.sc_sch.status)

        w_grps = [w_type, w_idx]

        rtol_decimals = int(np.abs(np.ceil(np.log10(rtol))))

        aggs_ = (
            ws_.group_by(w_grps)
            .agg(
                **{
                    ts: pl.first(t),
                    te: pl.last(t),
                    mx_a: pl.col(mxd).abs().sum() + 1,
                    unmx_a: pl.col(unmx).abs().sum() + 1,
                    mx_var: pl.col(mxd).abs().var(),
                    mx_u: pl.col(mxd).abs().mean(),
                }
            )
            .with_columns(
                **{
                    mx_fano: pl.col(mx_var) / pl.col(mx_u),
                    rc_sc: pl.col(unmx_a) / pl.col(mx_a),
                    rtol_l: pl.lit(rtol),
                }
            )
            .with_columns(**{tolchk: pl.col(rc_sc).sub(1).abs().round(rtol_decimals)})
            .with_columns(
                **{
                    tolpass: pl.col(tolchk) <= pl.col("rtol"),
                    u_fano: pl.col(mx_fano).filter(pl.col(w_type) == "peak").mean(),
                }
            )
            .with_columns(
                **{
                    div_fano: (pl.col(mx_fano) / pl.col(u_fano)).over(w_type),
                }
            )
            .with_columns(
                **{
                    fanopass: pl.col(div_fano) <= ftol,
                }
            )
            .with_columns(
                **{
                    status: pl.when(
                        (pl.col(w_type) == "peak") & (pl.col(tolpass) == True)
                    )
                    .then(pl.lit("valid"))
                    .when((pl.col(w_type) == "interpeak") & (pl.col(tolpass) == True))
                    .then(pl.lit("valid"))
                    .when(
                        (pl.col(w_type) == "interpeak") & (pl.col("fanopass") == True)
                    )
                    .then(pl.lit("needs review"))
                    .otherwise(pl.lit("invalid"))
                }
            )
            .join(self.grading, how="left", on="status")
            .join(self.grading_colors, how="left", on="status")
            .sort(w_type, w_idx)
        )

        aggs__ = aggs_.to_pandas()
        FitAssessScores.validate(aggs__, lazy=True)

        aggs = DataFrame[FitAssessScores](aggs__)
        return aggs

    def print_report_card(
        self,
        scores: DataFrame,
    ):
        """
        Assemble a series of strings for printing a final report
        """

        def gen_report_str(
            x: Series,
            recon_score_col: str,
            fano_factor_col: str,
            status_col: str,
            w_type_col: str,
            w_idx_col: str,
            t_start_col: str,
            t_end_col: str,
            grading: dict[str, str],
            grading_colors: dict[str, tuple[str]],
        ):
            columns = ["grading", "window", "time_range", "scores", "warning"]

            rs = pd.DataFrame(index=x.index, columns=columns)

            status = x.status.iloc[0]
            w_type = x[w_type_col].iloc[0]
            w_idx = x[w_idx_col].iloc[0]
            time_start = x[t_start_col].iloc[0]
            time_end = x[t_end_col].iloc[0]
            recon_score = x[recon_score_col].iloc[0]
            fano_factor = x[fano_factor_col].iloc[0]

            rs["grading"] = grading[status]

            rs["window"] = f"{w_type} window {w_idx}"
            rs["time_range"] = f"(t: {time_start} - {time_end})"
            rs["scores"] = f"R-score = {recon_score} & Fano Ratio = {fano_factor}\n"

            rs["warning"] = self.get_warning(status, w_idx)

            rs_list = rs.iloc[0].astype(str).tolist()
            report_str = " ".join(rs_list)
            c_report_str = termcolor.colored(
                report_str, *grading_colors[status], attrs=["bold"]
            )

            return c_report_str

        report_str = ""
        report_str += "-------------------Chromatogram Reconstruction Report Card----------------------\n\n"
        report_str += "Reconstruction of Peaks\n"
        report_str += "=======================\n\n"

        c_report_strs = scores.groupby([self.ws_sch.w_type, self.ws_sch.w_idx]).apply(
            gen_report_str,  # type: ignore
            warnings=warnings,
            **{
                "recon_score_col": "recon_score",
                "fano_factor_col": "mixed_fano",
                "status_col": "status",
                "w_type_col": self.ws_sch.w_type,
                "w_idx_col": self.ws_sch.w_idx,
                "t_start_col": "time_start",
                "t_end_col": "time_end",
            },
        )  # type: ignore
        c_report_strs.name = "report_strs"

        # join the peak strings followed by a interpeak subheader then the interpeak strings
        c_report_strs = c_report_strs.reset_index()
        c_report_str = report_str + "\n".join(
            c_report_strs.loc[
                c_report_strs[self.ws_sch.w_type] == "peak", "report_strs"
            ]
        )

        c_report_str += "\n\n"

        c_report_str += "Signal Reconstruction of Interpeak Windows\n"
        c_report_str += "=======================\n\n"

        c_report_str += "\n".join(
            c_report_strs.loc[
                c_report_strs[self.ws_sch.w_type] == "interpeak", "report_strs"
            ]
        )

        import os

        os.system("color")
        termcolor.cprint(c_report_str, end="\n\n")

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

        review_mask = (score_df["tolpass"] == False) | (score_df["fanopass"] == True)

        score_df["status"] = score_df["status"].mask(review_mask, "needs review")

        return score_df


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
