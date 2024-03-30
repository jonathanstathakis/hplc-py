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
from hplc_py.deconvolution import definitions as KeysFitReport

from hplc_py.deconvolution import schemas as dc_schs, definitions as dc_defs

import numpy as np
import pandera as pa
from pandera.typing import DataFrame

from hplc_py.deconvolution.fit_assessment_grading_tables import (
    get_grading_colors_frame,
    get_grading_frame,
)

from dataclasses import dataclass

@dataclass
class KeysFitReport:
    w_type: str = "w_type"
    w_idx: str = "w_idx"
    mixed: str = "mixed"
    recon: str = "recon"
    time: str = "time"
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


class Reporter:

    def __init__(self, data: DataFrame[dc_schs.ActiveSignal], key_time: str):
        
        self.data = data
        self.keys_fit_report = KeysFitReport(time=key_time)

    @pa.check_types
    def calc_fit_report(
        self,
        rtol: float = dc_defs.VAL_RTOL,
        ftol: float = dc_defs.VAL_FTOL,
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
        data_ = self.data.pipe(pl.from_pandas)

        if not isinstance(data_, pl.DataFrame):
            raise TypeError("Expected input data to be a polars DataFrame")

        rtol_decimals = int(np.abs(np.ceil(np.log10(rtol))))

        scores: DataFrame[dc_schs.FitAssessScores] = (
            data_
            # per each window
            .group_by([self.keys_fit_report.w_type, self.keys_fit_report.w_idx])
            .agg(
                # time start
                pl.first(self.keys_fit_report.time).alias(self.keys_fit_report.time_start),
                # time end
                pl.last(self.keys_fit_report.time).alias(self.keys_fit_report.time_end),
                # area mixed
                pl.col(self.keys_fit_report.mixed).abs().sum().add(1).alias(self.keys_fit_report.area_mixed),
                # area unmixed
                pl.col(self.keys_fit_report.recon).abs().sum().add(1).alias(self.keys_fit_report.area_unmixed),
                # variance mixed
                pl.col(self.keys_fit_report.mixed).abs().var().alias(self.keys_fit_report.var_mixed),
                # mean mixed
                pl.col(self.keys_fit_report.mixed).abs().mean().alias(self.keys_fit_report.mean_mixed),
            )
            .with_columns(
                # fano factor mixed
                pl.col(self.keys_fit_report.var_mixed).truediv(pl.col(self.keys_fit_report.mean_mixed)).alias(self.keys_fit_report.fano_mixed),
                # score recon
                pl.col(self.keys_fit_report.area_unmixed).truediv(pl.col(self.keys_fit_report.area_mixed)).alias(self.keys_fit_report.score_recon),
                # rtol
                pl.lit(rtol).alias(self.keys_fit_report.rtol),
            )
            .with_columns(
                # tolcheck
                pl.col(self.keys_fit_report.score_recon).sub(1).abs().round(rtol_decimals).alias(self.keys_fit_report.tolcheck)
            )
            .with_columns(
                # tolpass
                pl.col(self.keys_fit_report.tolcheck).le(pl.col(self.keys_fit_report.rtol)).alias(self.keys_fit_report.tolpass),
                # fano factor mixed mean
                pl.col(self.keys_fit_report.fano_mixed)
                .filter(pl.col(self.keys_fit_report.w_type).eq(self.keys_fit_report.w_type_peak))
                .mean()
                .alias(self.keys_fit_report.mean_fano),
            )
            .with_columns(
                # fano factor mixed div by its mean
                pl.col(self.keys_fit_report.fano_mixed)
                .truediv(pl.col(self.keys_fit_report.mean_fano))
                .over(self.keys_fit_report.w_type)
                .alias(self.keys_fit_report.fano_div),
            )
            .with_columns(
                # fano factor threshold pass
                pl.col(self.keys_fit_report.fano_div).le(ftol).alias(self.keys_fit_report.fanopass),
            )
            .with_columns(
                # status
                pl.when(
                    (pl.col(self.keys_fit_report.w_type).eq(self.keys_fit_report.w_type_peak)) & (pl.col(self.keys_fit_report.tolpass).eq(True))
                )
                .then(pl.lit(self.keys_fit_report.val_status_valid))
                .when(
                    (pl.col(self.keys_fit_report.w_type).eq(self.keys_fit_report.w_type_interpeak))
                    & (pl.col(self.keys_fit_report.tolpass).eq(True))
                )
                .then(pl.lit(self.keys_fit_report.val_status_valid))
                .when(
                    (pl.col(self.keys_fit_report.w_type).eq(self.keys_fit_report.w_type_interpeak))
                    & (pl.col(self.keys_fit_report.fanopass).eq(True))
                )
                .then(pl.lit(self.keys_fit_report.val_status_needs_review))
                .otherwise(pl.lit(self.keys_fit_report.val_status_invalid))
                .alias(self.keys_fit_report.status)
            )
            # grading
            .join(grading_frame, how="left", on=self.keys_fit_report.status)
            # grading color
            .join(grading_color_frame, how="left", on=self.keys_fit_report.status)
            .sort(self.keys_fit_report.time_start, self.keys_fit_report.w_idx)
            .to_pandas()
            .pipe(dc_schs.FitAssessScores.validate, lazy=True)
            .pipe(DataFrame[dc_schs.FitAssessScores])
        )  # fmt: skip

        return scores
