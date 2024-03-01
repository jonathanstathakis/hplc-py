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
from hplc_py.deconvolution import definitions as dc_defs
from hplc_py.map_windows import definitions as mw_defs

from hplc_py.deconvolution.schemas import (
    FitAssessScores,
)


import numpy as np

import pandera as pa
from pandera.typing import DataFrame

from hplc_py.deconvolution.fit_assessment_grading_tables import (
    get_grading_colors_frame,
    get_grading_frame,
)


@pa.check_types
def calc_fit_scores(
    X_w_with_recon: DataFrame[X_Windowed_With_Recon],
    rtol: float = dc_defs.VAL_RTOL,
    ftol: float = dc_defs.VAL_FTOL,
    grading_frame: pl.DataFrame = get_grading_frame(),
    grading_color_frame: pl.DataFrame = get_grading_colors_frame(),
) -> DataFrame[FitAssessScores]:
    """
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

    rtol_decimals = int(np.abs(np.ceil(np.log10(rtol))))
    scores: DataFrame[FitAssessScores] = (
        X_w_with_recon
        .pipe(pl.from_pandas)
        # per each window
        .group_by([dc_defs.W_TYPE, dc_defs.W_IDX])
        .agg(
            # time start
            pl.first(dc_defs.X_IDX).alias(dc_defs.KEY_TIME_START),
            # time end
            pl.last(dc_defs.X_IDX).alias(dc_defs.KEY_TIME_END),
            # area mixed
            pl.col(dc_defs.X).abs().sum().add(1).alias(dc_defs.KEY_AREA_MIXED),
            # area unmixed
            pl.col(dc_defs.KEY_RECON).abs().sum().add(1).alias(dc_defs.KEY_AREA_UNMIXED),
            # variance mixed
            pl.col(dc_defs.X).abs().var().alias(dc_defs.KEY_VAR_MIXED),
            # mean mixed
            pl.col(dc_defs.X).abs().mean().alias(dc_defs.KEY_MEAN_MIXED),
        )
        .with_columns(
            # fano factor mixed
            pl.col(dc_defs.KEY_VAR_MIXED).truediv(pl.col(dc_defs.KEY_MEAN_MIXED)).alias(dc_defs.KEY_FANO_MIXED),
            # score recon
            pl.col(dc_defs.KEY_AREA_UNMIXED).truediv(pl.col(dc_defs.KEY_AREA_MIXED)).alias(dc_defs.KEY_SCORE_RECON),
            # rtol
            pl.lit(rtol).alias(dc_defs.KEY_RTOL),
        )
        .with_columns(
            # tolcheck
            pl.col(dc_defs.KEY_SCORE_RECON).sub(1).abs().round(rtol_decimals).alias(dc_defs.KEY_TOLCHECK)
        )
        .with_columns(
            # tolpass
            pl.col(dc_defs.KEY_TOLCHECK).le(pl.col(dc_defs.KEY_RTOL)).alias(dc_defs.KEY_TOLPASS),
            # fano factor mixed mean
            pl.col(dc_defs.KEY_FANO_MIXED)
            .filter(pl.col(dc_defs.W_TYPE).eq(dc_defs.VAL_W_TYPE_PEAK))
            .mean()
            .alias(dc_defs.KEY_MEAN_FANO),
        )
        .with_columns(
            # fano factor mixed div by its mean
            pl.col(dc_defs.KEY_FANO_MIXED)
            .truediv(pl.col(dc_defs.KEY_MEAN_FANO))
            .over(dc_defs.W_TYPE)
            .alias(dc_defs.KEY_FANO_DIV),
        )
        .with_columns(
            # fano factor threshold pass
            pl.col(dc_defs.KEY_FANO_DIV).le(ftol).alias(dc_defs.KEY_FANOPASS),
        )
        .with_columns(
            # status
            pl.when(
                (pl.col(dc_defs.W_TYPE).eq(dc_defs.VAL_W_TYPE_PEAK)) & (pl.col(dc_defs.KEY_TOLPASS).eq(True))
            )
            .then(pl.lit(dc_defs.VAL_STATUS_VALID))
            .when(
                (pl.col(dc_defs.W_TYPE).eq(dc_defs.VAL_W_TYPE_INTERPEAK))
                & (pl.col(dc_defs.KEY_TOLPASS).eq(True))
            )
            .then(pl.lit(dc_defs.VAL_STATUS_VALID))
            .when(
                (pl.col(dc_defs.W_TYPE).eq(dc_defs.VAL_W_TYPE_INTERPEAK))
                & (pl.col(dc_defs.KEY_FANOPASS).eq(True))
            )
            .then(pl.lit(dc_defs.VAL_STATUS_NEEDS_REVIEW))
            .otherwise(pl.lit(dc_defs.VAL_STATUS_INVALID))
            .alias(dc_defs.KEY_STATUS)
        )
        # grading
        .join(grading_frame, how="left", on=dc_defs.KEY_STATUS)
        # grading color
        .join(grading_color_frame, how="left", on=dc_defs.KEY_STATUS)
        .sort(dc_defs.KEY_TIME_START, dc_defs.W_IDX)
        .to_pandas()
        .pipe(FitAssessScores.validate, lazy=True)
        .pipe(DataFrame[FitAssessScores])
    )  # fmt: skip

    return scores
