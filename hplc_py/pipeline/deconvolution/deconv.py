"""
Methods relevant to the deconvolution pipeline
"""

import pandera as pa
import polars as pl
from hplc_py.deconvolution import definitions as Keys, schemas as dc_schs
from hplc_py.map_peaks import schemas as mp_schs
from hplc_py.map_peaks import definitions as mp_defs
from pandera.typing import DataFrame

from hplc_py.map_windows import definitions as mw_defs


def prepare_peak_msnts_windowed(
    peak_map: dict,
    X_w: DataFrame[dc_schs.X_Windowed],
) -> DataFrame[dc_schs.PeakMsntsWindowed]:
    """
    Prepares a table OptParamPeakInput containing the peak maxima location, amplitude and peak WHH.
    """

    maxima = peak_map["maxima"].pipe(pl.from_pandas)

    split_msnt_widths = (
        peak_map["widths"]
        .pipe(pl.from_pandas)
        .with_columns(
            pl.col(mp_defs.KEY_MSNT)
            .str.split("_")
            .list.to_struct()
            .struct.rename_fields([mp_defs.DIM, mp_defs.LOC]),
            pl.col(mp_defs.VALUE),
        )
        .unnest(mp_defs.KEY_MSNT)
        .select(
            pl.col(mp_defs.P_IDX),
            pl.col(mp_defs.LOC),
            pl.col(mp_defs.DIM),
            pl.col(mp_defs.VALUE),
        )
    )
    whh = split_msnt_widths.filter(pl.col(mp_defs.LOC) == mp_defs.KEY_WHH).select(
        pl.col(mp_defs.P_IDX),
        pl.col(mp_defs.LOC).alias("param"),
        pl.col(mp_defs.DIM),
        pl.col(mp_defs.VALUE),
    )

    peak_msnts = (
        pl.concat([maxima, whh]).to_pandas().pipe(dc_schs.PeakMsnts.validate, lazy=True)
    )

    breakpoint()
    peak_msnts_windowed: DataFrame[dc_schs.PeakMsntsWindowed] = window_peak_map(
        peak_msnts=peak_msnts,
        X_w=X_w,
    )

    return peak_msnts_windowed


@pa.check_types(lazy=True)
def window_peak_map(
    peak_msnts: DataFrame[dc_schs.PeakMsnts],
    X_w: DataFrame[dc_schs.X_Windowed],
) -> DataFrame[dc_schs.PeakMsntsWindowed]:
    """
    add w_idx to to peak map for later lookups
    """

    # filter whh_and_maxima to dim == X_idx then join with X_w to find the window of each peak

    windowed_peaks = (
        peak_msnts.pipe(pl.from_pandas)
        .filter(pl.col(mp_defs.DIM) == mp_defs.X_IDX)
        .select(pl.col(mp_defs.P_IDX), pl.col(mp_defs.VALUE).cast(int))
        .join(
            X_w.pipe(pl.from_pandas).select([Keys.W_TYPE, Keys.W_IDX, Keys.X_IDX]),
            how="left",
            left_on=mp_defs.VALUE,
            right_on=mp_defs.X_IDX,
        )
        .select(pl.col([mp_defs.P_IDX, mw_defs.W_TYPE, mw_defs.W_IDX]))
    )

    # join the intermediate table with whh_and_maxima on p_idx

    windowed_whh_and_maxima = (
        peak_msnts.pipe(pl.from_pandas)
        .join(
            windowed_peaks,
            on=[mp_defs.P_IDX],
            how="left",
        )
        .select(
            pl.col(
                [
                    mw_defs.W_TYPE,
                    mw_defs.W_IDX,
                    mp_defs.P_IDX,
                    mp_defs.LOC,
                    mp_defs.DIM,
                    mp_defs.VALUE,
                ]
            )
        )
        .to_pandas()
        .pipe(dc_schs.PeakMsntsWindowed.validate, lazy=True)
        .pipe(DataFrame[dc_schs.PeakMsntsWindowed])
    )

    return windowed_whh_and_maxima
