from pandera.typing import DataFrame
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes as Axes

from ..common_schemas import X_Schema
from ..map_peaks.viz import UI_PlotPeakMapWide
from ..map_peaks.schemas import PeakMapWide
from .schemas import X_Windowed
from typing import Self
import pandera as pa
class UI_WindowMapViz(UI_PlotPeakMapWide):
    
    @pa.check_types
    def __init__(
        self,
        peak_map: DataFrame[PeakMapWide],
        X_w: DataFrame[X_Windowed],
        ax: Axes = plt.gca(),
    ):
        self.w_type_key = 'w_type'
        self.w_idx_key = 'w_idx'
        self.X_idx_key = 'X_idx'
        self.X_key = 'X'
        
        X = DataFrame[X_Schema](X_w[[self.X_idx_key, self.X_key]])
        
        super().__init__(X=X, peak_map=peak_map, ax=ax)
        
        self.X_w = X_w
        
        
    def draw_peak_windows(
        self,
    )->Self:
        draw_peak_windows(
            X_w=pl.from_pandas(self.X_w),
            w_type_key=self.w_type_key,
            w_idx_key=self.w_idx_key,
            X_idx_key=self.X_idx_key,
            ax=self.ax,
        )
        return self


def draw_peak_windows(
    X_w: pl.DataFrame,
    w_type_key: str,
    w_idx_key: str,
    X_idx_key: str,
    ax: Axes,
):
    """
    Plot each window as a Rectangle

    height is the maxima of the signal.

    """
    if not isinstance(X_w, pl.DataFrame):
        raise TypeError("expected ws to be a polars dataframe")

    window_bounds = find_window_bounds(
        X_w=X_w, w_type_key=w_type_key, w_idx_key=w_idx_key, X_idx_key=X_idx_key
    )

    peak_window_bounds: pl.DataFrame = window_bounds.filter(pl.col("w_type") == "peak")

    # assign colors

    # handles, labels = ax.get_legend_handles_labels()
    import seaborn as sns

    w_idx_unique = peak_window_bounds.select("w_idx").unique(maintain_order=True)

    colors = sns.color_palette("deep", n_colors = len
                               (w_idx_unique))
    
    peak_window_bounds_colored = (
        peak_window_bounds
        .with_columns(color=pl.Series(colors))
    )
    

    grpby_obj = peak_window_bounds_colored.group_by(
        [w_idx_key], maintain_order=True
    )

    for label, grp in grpby_obj:
        x0 = grp.item(0, "start")
        x1 = grp.item(0, "end")

        # testing axvspan

        ax.axvspan(
            x0,
            x1,
            label=f"peak window {label}",
            color=grp.item(0, 'color'),
            alpha=0.25,
        )


def find_window_bounds(
    X_w: pl.DataFrame,
    w_type_key: str,
    w_idx_key: str,
    X_idx_key: str,
):
    
    window_bounds = (
        X_w.group_by([w_type_key, w_idx_key])
        .agg(
            start=pl.col(str(X_idx_key)).first(),
            end=pl.col(str(X_idx_key)).last(),
        )
        .sort("start")
    )

    return window_bounds
