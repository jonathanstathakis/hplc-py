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
from hplc_py.io_validation import check_input_is_float
from hplc_py.hplc_py_typing.hplc_py_typing import FitAssessScores, WindowedSignal, PSignals

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


class WindowedMixed(pa.DataFrameModel):
    pass


class WindowedUnmixed(pa.DataFrameModel):
    pass


@dataclass
class FitAssessment:
    @pa.check_types
    def assess_fit(
        self,
        ws: DataFrame[WindowedSignal],
        psignals: DataFrame[PSignals],
        rtol: float = 1e-2,
        ftol: float = 1e-2,
    ):
        score_df = self.calc_wdw_aggs(ws, psignals, rtol, ftol)

        self.print_report_card(score_df)
        return score_df

    @pa.check_types
    def calc_wdw_aggs(
        self,
        ws: DataFrame[WindowedSignal],
        psignals: DataFrame[PSignals],
        rtol: float,
        ftol: float,
    )->DataFrame[FitAssessScores]:
        
        check_input_is_float(rtol)
        check_input_is_float(ftol)
        
        import polars as pl

        rs_: pl.DataFrame = pl.from_pandas(psignals)
        ws_: pl.DataFrame = pl.from_pandas(ws)

        recon_sig = rs_.pivot(
            columns="p_idx", index=["time_idx", "time"], values="amp_unmixed"
        ).select(
            pl.col("time_idx", "time"),
            pl.sum_horizontal(pl.exclude(["time_idx", "time"])).alias("amp_unmixed"),
        )

        wrs = recon_sig.join(
            ws_.select(["w_type", "w_idx", "time_idx", "amp"]).rename(
                {"amp": "amp_mixed"}
            ),
            on="time_idx",
            how="left",
        ).select(["w_type", "w_idx", "time_idx", "time", "amp_mixed", "amp_unmixed"])

        w_grps = ["w_type", "w_idx"]

        rtol_decimals = int(np.abs(np.ceil(np.log10(rtol))))

        aggs_ = (
            wrs.group_by(w_grps)
            .agg(
                time_start=pl.first("time"),
                time_end=pl.last("time"),
                signal_area=pl.col("amp_mixed").abs().sum() + 1,
                inferred_area=pl.col("amp_unmixed").abs().sum() + 1,
                mixed_var=pl.col("amp_mixed").abs().var(),
                mixed_mean=pl.col("amp_mixed").abs().mean(),
            )
            .with_columns(
                mixed_fano=pl.col("mixed_var") / pl.col("mixed_mean"),
                recon_score=pl.col("inferred_area") / pl.col("signal_area"),
                rtol=pl.lit(rtol),
            )
            .with_columns(
                tolcheck=pl.col("recon_score").sub(1).abs().round(rtol_decimals)
            )
            .with_columns(
                tolpass=pl.col("tolcheck") <= pl.col("rtol"),
                u_peak_fano=pl.col("mixed_fano")
                .filter(pl.col("w_type") == "peak")
                .mean(),
            )
            .with_columns(
                fano_div=(pl.col("mixed_fano") / pl.col("u_peak_fano")).over("w_type"),
            )
            .with_columns(
                fanopass=pl.col("fano_div") <= ftol,
            )
            .with_columns(
                status=pl.when(
                    (pl.col("w_type") == "peak") & (pl.col("tolpass") == True)
                )
                .then(pl.lit("valid"))
                .when((pl.col("w_type") == "interpeak") & (pl.col("tolpass") == True))
                .then(pl.lit("valid"))
                .when((pl.col("w_type") == "interpeak") & (pl.col("fanopass") == True))
                .then(pl.lit("needs review"))
                .otherwise(pl.lit("invalid"))
            )
            .sort("w_type", "w_idx")
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

        grading = {
            "valid": "A+ Success: ",
            "invalid": "F, Failed: ",
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
            return f"Interpeak window {w_idx} is not well reconstructed by mixture and has an appreciable Fano factor compared to peak region(s). This suggests you have missed a peak in this region. Consider adding manual peak positioning by passing `known_peaks` to `fit_peaks()`"

        def valid_warning(w_idx: int):
            return ""

        warnings = {
            "valid": valid_warning,
            "invalid": invalid_warning,
            "needs review": review_warning,
        }

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
            warnings: dict[str, Callable],
        ):
            columns = ["grading", "window", "time_range", "scores", "warning"]

            rs = pd.DataFrame(index=x.index, columns=columns)

            rs["grading"] = grading[x.status.iloc[0]]
            rs["window"] = f"{x[w_type_col].iloc[0]} window  {x[w_idx_col].iloc[0]}"
            rs[
                "time_range"
            ] = f"(t: {x[t_start_col].iloc[0]} - {x[t_end_col].iloc[0]})"
            rs[
                "scores"
            ] = f"R-score = {x[recon_score_col].iloc[0]} & Fano Ratio = {x[fano_factor_col].iloc[0]}\n"
            rs["warning"] = warnings[x[status_col].iloc[0]](x[w_idx_col].iloc[0])

            rs_list = rs.iloc[0].astype(str).tolist()
            report_str = " ".join(rs_list)
            c_report_str = termcolor.colored(
                report_str, *grading_colors[x[status_col].iloc[0]], attrs=["bold"]
            )

            return c_report_str

        report_str = ""
        report_str += "-------------------Chromatogram Reconstruction Report Card----------------------\n\n"
        report_str += "Reconstruction of Peaks\n"
        report_str += "=======================\n\n"

        c_report_strs = scores.groupby(["w_type", "w_idx"]).apply(
            gen_report_str, #type: ignore
            grading=grading,
            grading_colors=grading_colors,
            warnings=warnings,
            **{"recon_score_col": "recon_score",
             "fano_factor_col": "mixed_fano",
            "status_col": "status",
            "w_type_col": "w_type",
            "w_idx_col": "w_idx",
            "t_start_col": "time_start",
            "t_end_col": "time_end",
             },
        ) #type: ignore
        c_report_strs.name = "report_strs"

        # join the peak strings followed by a interpeak subheader then the interpeak strings
        c_report_strs = c_report_strs.reset_index()
        c_report_str = report_str + "\n".join(
            c_report_strs.loc[c_report_strs["w_type"] == "peak", "report_strs"]
        )

        c_report_str += "\n\n"

        c_report_str += "Signal Reconstruction of Interpeak Windows\n"
        c_report_str += "=======================\n\n"

        c_report_str += "\n".join(
            c_report_strs.loc[c_report_strs["w_type"] == "interpeak", "report_strs"]
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
