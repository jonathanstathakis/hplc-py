from hplc_py.common import definitions as com_defs
from hplc_py.common.common_schemas import TimeWindowMapping
from hplc_py.map_signal.map_windows.viz_hv import WindowMapViz
from hplc_py.map_signal.map_windows.definitions import W_IDX, W_TYPE
from hplc_py.map_signal.map_windows.schemas import WindowBounds, X_Windowed
from hplc_py import precision
import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class WindowMap(precision.Precision):
    @pa.check_types
    def __init__(self, X_windowed: DataFrame[X_Windowed]):

        self._X_windowed: DataFrame[X_Windowed] = X_windowed

    @property
    def X_windowed(self):
        return self._X_windowed

    @X_windowed.getter
    def X_windowed(self):
        return self._X_windowed.with_columns(pl.col("X").round(self._precision))

    @property
    def window_bounds(self) -> DataFrame[WindowBounds]:

        bounds: DataFrame[WindowBounds] = (
            self.X_windowed.group_by([W_TYPE, W_IDX], maintain_order=True)
            .agg(
                pl.col(com_defs.IDX).first().alias("left"),
                pl.col(com_defs.IDX).last().alias("right"),
            )
            .pipe(WindowBounds.validate, lazy=True)
            .pipe(DataFrame[WindowBounds])
        )

        return bounds

    @property
    def X_maxima(self) -> float:
        return self.X_windowed.select("X").max().item()

    @property
    def peak_windows(self):
        return self.window_bounds.filter(pl.col("w_type").eq("peak")).drop("w_type")

    @property
    def peak_windows_long(self):
        return self.peak_windows.melt(
            id_vars="w_idx",
            value_vars=["left", "right"],
            value_name="idx",
            variable_name="side",
        ).sort("w_idx", descending=False)

    @property
    def peak_window_left_bounds(self):
        return self.peak_windows.select("left").to_series()

    @property
    def peak_window_right_bounds(self):
        return self.peak_windows.select("right").to_series()

    @property
    def interpeak_windows(self):
        return self.window_bounds.filter(pl.col("w_type").eq("interpeak")).drop(
            "w_type"
        )

    @property
    def interpeak_window_left_bounds(self):
        return self.interpeak_windows.select("left").to_series()

    @property
    def interpeak_window_right_bounds(self):
        return self.interpeak_windows.select("right").to_series()

    def as_viz(self):
        return WindowMapViz(self)

    @property
    def time_window_mapping(self) -> TimeWindowMapping:
        return self.X_windowed.select(
            pl.col("w_type"), pl.col("w_idx"), pl.col("idx")
        ).pipe(DataFrame[TimeWindowMapping])

    def __repr__(self):
        return f"contains the following tables:\n\tX_windowed\n\t\tcolumns: {self.X_windowed.columns}\n\t\tshape:{self.X_windowed.shape}\n\twindow_bounds:\n\t\tcolumns: {self.window_bounds.columns}\n\t\tshape: {self.window_bounds.shape}"