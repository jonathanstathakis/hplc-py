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
from hplc_py.deconvolution.schemas import X_Windowed_With_Recon
from hplc_py.deconvolution import definitions as Keys
from hplc_py.map_windows import definitions as mw_defs

from hplc_py.deconvolution import schemas as dc_schs

import numpy as np
import pandera as pa
from pandera.typing import DataFrame

from hplc_py.deconvolution.fit_assessment_grading_tables import (
    get_grading_colors_frame,
    get_grading_frame,
)


@pa.check_types
def calc_fit_report(
    data: DataFrame[dc_schs.ActiveSignal],
    rtol: float = Keys.VAL_RTOL,
    ftol: float = Keys.VAL_FTOL,
    grading_frame: pl.DataFrame = get_grading_frame(),
    grading_color_frame: pl.DataFrame = get_grading_colors_frame(),
) -> DataFrame[dc_schs.FitAssessScores]:
    """
    :param data: a polars dataframe with columns: "w_type", "w_idx", "x", "mixed", "unmixed"

    # Notes

    1. Recon Score

    Use to gauge the reconstruction of peak regions.

    .. math::
    R=\\frac{1+\\text{AUC}^\\text{(unmixed)}}{1+\\text{AUC}^\\text{(mixed)}}

    The ratio between reconstruction scores adjusted by adding each to 1 to scale the score to a meaningful level - deviations in very small peaks would have an outsized effect relative to the overall amplitude range.

    where 1 is a perfect reconstruction.

    2. Fano Factor

    use to gauge the reconstruction of interpeak regions.

    F(t)=\\frac{\\sigma^2_t}{\\mu_t}



    A small fano factor indicates background noise, a large fano factor indicates an undetected peak.

    fano factor is like the coefficient of variation..
    measures the dispersion of a counting process..

    \mu_t is the mean number of events of a counting process.


    """
    data_ = data.pipe(pl.from_pandas)

    if not isinstance(data_, pl.DataFrame):
        raise TypeError("Expected input data to be a polars DataFrame")

    from dataclasses import dataclass

    @dataclass
    class Keys:
        w_type: str = "w_type"
        w_idx: str = "w_idx"
        mixed: str = "mixed"
        recon: str = "recon"
        x: str = "x"
        time_start: str = "time_start"
        time_end: str = "time_end"
        area_mixed: str = "area_mixed"
        area_unmixed: str = "area_unmixed"
        var_unmixed: str = "var_unmixed"
        var_mixed: str = "var_mixed"
        mean_mixed: str = "mean_mixed"
        fano_mixed: str = "fano_mixed"
        mean_fano: str = "fano_mean"
        score_recon: str = "score_recon"
        tolcheck: str = "tolcheck"
        tolpass: str = "tolpass"
        w_type_peak: str = "peak"
        w_type_interpeak: str = "interpeak"
        fano_div: str = "div_fano"
        fanopass: str = "fanopass"
        status: str = "status"
        rtol: str = "rtol"
        val_status_valid: str = "valid"
        val_status_needs_review: str = "needs review"
        val_status_invalid: str = "invalid"

    rtol_decimals = int(np.abs(np.ceil(np.log10(rtol))))

    scores: DataFrame[dc_schs.FitAssessScores] = (
        data_
        # per each window
        .group_by([Keys.w_type, Keys.w_idx])
        .agg(
            # time start
            pl.first(Keys.x).alias(Keys.time_start),
            # time end
            pl.last(Keys.x).alias(Keys.time_end),
            # area mixed
            pl.col(Keys.mixed).abs().sum().add(1).alias(Keys.area_mixed),
            # area unmixed
            pl.col(Keys.recon).abs().sum().add(1).alias(Keys.area_unmixed),
            # variance mixed
            pl.col(Keys.mixed).abs().var().alias(Keys.var_mixed),
            # mean mixed
            pl.col(Keys.mixed).abs().mean().alias(Keys.mean_mixed),
        )
        .with_columns(
            # fano factor mixed
            pl.col(Keys.var_mixed).truediv(pl.col(Keys.mean_mixed)).alias(Keys.fano_mixed),
            # score recon
            pl.col(Keys.area_unmixed).truediv(pl.col(Keys.area_mixed)).alias(Keys.score_recon),
            # rtol
            pl.lit(rtol).alias(Keys.rtol),
        )
        .with_columns(
            # tolcheck
            pl.col(Keys.score_recon).sub(1).abs().round(rtol_decimals).alias(Keys.tolcheck)
        )
        .with_columns(
            # tolpass
            pl.col(Keys.tolcheck).le(pl.col(Keys.rtol)).alias(Keys.tolpass),
            # fano factor mixed mean
            pl.col(Keys.fano_mixed)
            .filter(pl.col(Keys.w_type).eq(Keys.w_type_peak))
            .mean()
            .alias(Keys.mean_fano),
        )
        .with_columns(
            # fano factor mixed div by its mean
            pl.col(Keys.fano_mixed)
            .truediv(pl.col(Keys.mean_fano))
            .over(Keys.w_type)
            .alias(Keys.fano_div),
        )
        .with_columns(
            # fano factor threshold pass
            pl.col(Keys.fano_div).le(ftol).alias(Keys.fanopass),
        )
        .with_columns(
            # status
            pl.when(
                (pl.col(Keys.w_type).eq(Keys.w_type_peak)) & (pl.col(Keys.tolpass).eq(True))
            )
            .then(pl.lit(Keys.val_status_valid))
            .when(
                (pl.col(Keys.w_type).eq(Keys.w_type_interpeak))
                & (pl.col(Keys.tolpass).eq(True))
            )
            .then(pl.lit(Keys.val_status_valid))
            .when(
                (pl.col(Keys.w_type).eq(Keys.w_type_interpeak))
                & (pl.col(Keys.fanopass).eq(True))
            )
            .then(pl.lit(Keys.val_status_needs_review))
            .otherwise(pl.lit(Keys.val_status_invalid))
            .alias(Keys.status)
        )
        # grading
        .join(grading_frame, how="left", on=Keys.status)
        # grading color
        .join(grading_color_frame, how="left", on=Keys.status)
        .sort(Keys.time_start, Keys.w_idx)
        .to_pandas()
        .pipe(dc_schs.FitAssessScores.validate, lazy=True)
        .pipe(DataFrame[dc_schs.FitAssessScores])
    )  # fmt: skip

    return scores
