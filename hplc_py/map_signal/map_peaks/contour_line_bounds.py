from hplc_py.common.common_schemas import X_Schema
from hplc_py.map_signal.map_peaks import definitions as mp_defs, schemas as mp_schs
from hplc_py.map_signal.map_peaks import viz_matplotlib as mp_viz

from hplc_py.precision import Precision
import polars as pl
from pandera.typing.polars import DataFrame as Polars_DataFrame


class ContourLineBounds(Precision):
    """
    _summary_

    :param Precision: _description_
    :type Precision: _type_
    """
    def __init__(
        self,
        pm_melt: pl.DataFrame,
        X: Polars_DataFrame[X_Schema],
    ):
        """
        _summary_

        :param pm_melt: _description_
        :type pm_melt: pl.DataFrame
        :param X: _description_
        :type X: Polars_DataFrame[X_Schema]
        """
        super().__init__()
        self.contour_line_keys = [
            mp_defs.KEY_LEFT_PROM,
            mp_defs.KEY_RIGHT_PROM,
            mp_defs.KEY_LEFT_WHH,
            mp_defs.KEY_RIGHT_WHH,
            mp_defs.KEY_LEFT_PB,
            mp_defs.KEY_RIGHT_PB,
        ]

        # a normalised table of the contour bound measurements, prominence, whh, bases.

        peak_contours = pm_melt.filter(
            pl.col(mp_defs.KEY_MSNT).is_in(self.contour_line_keys)
        )

        contour_bounds_split = (
            peak_contours.with_columns(
                pl.col(mp_defs.KEY_MSNT)
                .str.split("_")
                .list.to_struct(n_field_strategy="max_width")
                .alias("msnt_split")
            )
            .drop(mp_defs.KEY_MSNT)
            .unnest("msnt_split")
            .rename(
                {
                    "field_0": mp_defs.LOC,
                    "field_1": mp_defs.KEY_MSNT,
                    "value": "X_idx_output",
                }
            )
        )

        countour_bounds_with_rounded_X_idx = contour_bounds_split.with_columns(
            pl.col("X_idx_output").round(0).cast(int).alias(mp_defs.KEY_IDX_ROUNDED)
        )

        contour_bounds_join_X = countour_bounds_with_rounded_X_idx.join(
            X,
            left_on=mp_defs.KEY_IDX_ROUNDED,
            right_on=mp_defs.IDX,
            how="left",
        )

        self._bounds = (
            contour_bounds_join_X
            .melt(
                id_vars=[mp_defs.P_IDX,mp_defs.LOC,mp_defs.KEY_MSNT],
                value_vars=[mp_defs.KEY_IDX_ROUNDED,mp_defs.X],
                variable_name=mp_defs.DIM,
                value_name=mp_defs.VALUE
                )
            .sort([mp_defs.P_IDX, mp_defs.KEY_MSNT, mp_defs.LOC, mp_defs.DIM])
            .pipe(mp_schs.ContourLineBoundsSchema.validate, lazy=True)
            .pipe(Polars_DataFrame[mp_schs.ContourLineBoundsSchema])
        )  # fmt: skip

    @property
    def bounds(self):
        """
        return the contour_line_bounds table. columns: ['p_idx':int, 'loc': str,'msnt': str, 'dim': sr,'value': float]
        """
        return self._bounds

    @bounds.getter
    def bounds(self):
        return self._bounds.with_columns(pl.col("value").round(self._precision))

    def get_base_side_as_series(self, side: str, msnt: str, unit: str) -> pl.Series:
        """
        Translate the countour_line_bounds to the left or right bases expected by WindowMapper
        """
        base: pl.Series = (
            self._bounds.filter(
                pl.col("msnt") == msnt,
                pl.col("dim") == unit,
                pl.col("loc") == side,
            )
            .select("value")
            .to_series()
            .rename(f"{msnt}_{side}")
        )

        return base

    def get_bound_by_type(self, msnt: str = "pb"):
        """
        return the contour line bounds correspnding to the msnt

        :base: choose from "pb", "whh", "prom", "all", or "none".
        """

        return self._bounds.filter(pl.col("msnt").eq(msnt))