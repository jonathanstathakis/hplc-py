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

import polars as pl
from hplc_py.io_validation import IOValid
from tests.tests_jonathan.tests.conftest import X_idx
from .schemas import PSignals

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

def get_grading_frame(
    status_key: str,
    grade_key: str,
    )->pl.DataFrame:
    grading = pl.DataFrame(
            {
                status_key: ["valid", "invalid", "needs review"],
                grade_key: ["A+, success", "F, failed", "C-, needs review"],
            }
        )
    return grading

def get_grading_colors_frame(status_key: str, color_key: str)->pl.DataFrame:
    grading_colors = pl.DataFrame(
            {
                status_key: ["valid", "invalid", "needs review"],
                color_key: [
                    "black, on_green",
                    "black, on_red",
                    "black, on_yellow",
                ],
            }
        )
    
    return grading_colors


class FitAssessment(IOValid):
    def __init__(self):
        self.ws_sch = X_Windowed
        self.sc_sch = FitAssessScores

    @pa.check_types
    def assess_fit(
        self,
        ws: DataFrame[X_Windowed],
        rtol: float = 1e-2,
        ftol: float = 1e-2,
    ):
        score_df = calc_wdw_aggs(
            ws,
            rtol,
            ftol)

        # self.print_report_card(score_df)
        return score_df

@pa.check_types
def calc_wdw_aggs(
    ws: DataFrame[X_Windowed],
    rtol: float,
    ftol: float,
    w_type_key: str,
    w_idx_key: str,
    time_start_key: str,
    X_idx_key: str,
    time_end_key: str,
    mixed_area_key: str,
    unmixed_area_key: str,
    mixed_var_key: str,
    unmixed_var_key: str,
    X_key: str,
    unmixed_key: str,
    mixed_mean_key: str,
    mixed_fano_key: str,
    recon_score_key: str,
    rtol_key: str,
    tolcheck_key: str,
    tolpass_key: str,
    mean_fano_key: str,
    peak_val: str,
    div_fano_key: str,
    fanopass_key: str,
    status_key: str,
    grading_frame: pl.DataFrame,
    grading_color_frame: pl.DataFrame,
    w_type_peak_val: str,
    w_type_interpeak_val: str,
    status_valid_val: str,
    status_needs_review_val: str,
    status_invalid_val: str,
    
) -> DataFrame[FitAssessScores]:

    ws_: pl.DataFrame = pl.from_pandas(ws)

    # declare the column labels from the schemas to improve readability of following
    # method chain. Allows final names to be declared in the Schema rather than locally.

    w_grps = [w_idx_key, w_type_key]

    rtol_decimals = int(np.abs(np.ceil(np.log10(rtol))))

    aggs_ = (
        ws_.group_by(w_grps)
        .agg(
            **{
                time_start_key: pl.first(X_idx_key),
                time_end_key: pl.last(X_idx_key),
                mixed_area_key: pl.col(X_key).abs().sum() + 1,
                unmixed_area_key: pl.col(unmixed_key).abs().sum() + 1,
                mixed_var_key: pl.col(unmixed_key).abs().var(),
                mixed_mean_key: pl.col(X_key).abs().mean(),
            }
        )
        .with_columns(
            **{
                mixed_fano_key: pl.col(mixed_var_key) / pl.col(mixed_mean_key),
                recon_score_key: pl.col(unmixed_area_key) / pl.col(mixed_area_key),
                rtol_key: pl.lit(rtol),
            }
        )
        .with_columns(**{tolcheck_key: pl.col(recon_score_key).sub(1).abs().round(rtol_decimals)})
        .with_columns(
            **{
                tolpass_key: pl.col(tolcheck_key) <= pl.col(rtol_key),
                mean_fano_key: pl.col(mixed_fano_key).filter(pl.col(w_type_key) == peak_val).mean(),
            }
        )
        .with_columns(
            **{
                div_fano_key: (pl.col(mixed_fano_key) / pl.col(mean_fano_key)).over(w_type_key),
            }
        )
        .with_columns(
            **{
                fanopass_key: pl.col(div_fano_key) <= ftol,
            }
        )
        .with_columns(
            **{
                status_key: pl.when(
                    (pl.col(w_type_key) == w_type_peak_val) & (pl.col(tolpass_key) == True)
                )
                .then(pl.lit(status_valid_val))
                .when((pl.col(w_type_key) == w_type_interpeak_val) & (pl.col(tolpass_key) == True))
                .then(pl.lit(status_valid_val))
                .when(
                    (pl.col(w_type_key) == w_type_interpeak_val) & (pl.col(fanopass_key) == True)
                )
                .then(pl.lit(status_needs_review_val))
                .otherwise(pl.lit(status_invalid_val))
            }
        )
        .join(grading_frame, how="left", on=status_key)
        .join(grading_color_frame, how="left", on=status_key)
        .sort(w_type_key, w_idx_key)
    )

    aggs__ = aggs_.to_pandas()
    FitAssessScores.validate(aggs__, lazy=True)

    aggs = DataFrame[FitAssessScores](aggs__)
    return aggs


def print_report_card(
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

        rs["warning"] = get_warning(status, w_idx)

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

    c_report_strs = scores.groupby([w_type_key, w_idx_key]).apply(
        gen_report_str,  # type: ignore
        warnings=warnings,
        **{
            "recon_score_col": "recon_score",
            "fano_factor_col": "mixed_fano",
            "status_col": "status",
            "w_type_col": w_type_key,
            "w_idx_col": w_idx_key,
            "t_start_col": "time_start",
            "t_end_col": "time_end",
        },
    )  # type: ignore
    c_report_strs.name = "report_strs"

    # join the peak strings followed by a interpeak subheader then the interpeak strings
    c_report_strs = c_report_strs.reset_index()
    c_report_str = report_str + "\n".join(
        c_report_strs.loc[c_report_strs[w_type_key] == "peak", "report_strs"]
    )

    c_report_str += "\n\n"

    c_report_str += "Signal Reconstruction of Interpeak Windows\n"
    c_report_str += "=======================\n\n"

    c_report_str += "\n".join(
        c_report_strs.loc[c_report_strs[w_type_key] == "interpeak", "report_strs"]
    )

    import os

    os.system("color")
    termcolor.cprint(c_report_str, end="\n\n")

    return c_report_str


def assign_status(
    score_df: DataFrame,
    status_key: str,
    invalid_val: str,
    valid_val: str,
    tolpass_key: str,
    fanopass_key: str,
    needs_review_val: str,
):
    """
    assign status of window based on tolpass and fanopass. if tolpass is invalid but fanopass is valid, assign 'needs review', else 'invalid'

    Treat the peaks and nonpeaks differently. Peaks are valid if tolpass is valid, nonpeaks are valid if tolpass valid, but need review
    """

    score_df[status_key] = invalid_val

    valid_mask = score_df[tolpass_key]

    # check fof tolpass validity
    score_df[status_key] = score_df[status_key].mask(valid_mask == True, valid_val)

    review_mask = (score_df[tolpass_key] == False) | (score_df[fanopass_key] == True)

    score_df[status_key] = score_df[status_key].mask(review_mask, needs_review_val)

    return score_df
