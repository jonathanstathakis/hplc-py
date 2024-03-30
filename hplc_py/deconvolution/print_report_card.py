from hplc_py.deconvolution import definitions as KeysFitReport


import pandas as pd
import termcolor
from pandera.typing import DataFrame, Series


import os


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

    c_report_strs = scores.groupby([KeysFitReport.W_TYPE, KeysFitReport.W_IDX]).apply(
        gen_report_str,  # type: ignore
        warnings=warnings,
        **{
            "recon_score_col": "recon_score",
            "fano_factor_col": "mixed_fano",
            "status_col": "status",
            "w_type_col": KeysFitReport.W_TYPE,
            "w_idx_col": KeysFitReport.W_IDX,
            "t_start_col": "time_start",
            "t_end_col": "time_end",
        },
    )  # type: ignore
    c_report_strs.name = "report_strs"

    # join the peak strings followed by a interpeak subheader then the interpeak strings
    c_report_strs = c_report_strs.reset_index()
    c_report_str = report_str + "\n".join(
        c_report_strs.loc[c_report_strs[KeysFitReport.W_TYPE] == "peak", "report_strs"]
    )

    c_report_str += "\n\n"

    c_report_str += "Signal Reconstruction of Interpeak Windows\n"
    c_report_str += "=======================\n\n"

    c_report_str += "\n".join(
        c_report_strs.loc[
            c_report_strs[KeysFitReport.W_TYPE] == "interpeak", "report_strs"
        ]
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
