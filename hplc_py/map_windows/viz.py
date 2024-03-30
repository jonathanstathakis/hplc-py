from pandera.typing import DataFrame
import polars as pl
from matplotlib.axes import Axes as Axes
from hplc_py.common import definitions as com_defs

from hplc_py.map_peaks.definitions import P_IDX
from hplc_py.map_windows import definitions as mw_defs
from hplc_py.map_windows.definitions import KEY_END, KEY_START, W_IDX, W_TYPE

from .schemas import X_Windowed
import pandera as pa
import holoviews as hv


def get_window_bounds(df: pl.DataFrame):
    """
    A convenience method to display the window bounds. For testing, will delete or
    recycle later.
    """
    bounds = df.group_by([W_TYPE, W_IDX], maintain_order=True).agg(
        pl.col(com_defs.X_IDX).first().alias(KEY_START),
        end=pl.col(com_defs.X_IDX).last().alias(KEY_END),
    )

    return bounds


class WindowMapViz:

    @pa.check_types
    def __init__(
        self,
        X_w: DataFrame[X_Windowed],
    ):
        self.X_w = X_w

    def draw_peak_windows(
        self,
        show=True,
    ):
        """
        draw peak windows as vspans, returning a holoviews plot object. intended to be
        combined with other plot objects to produce a complete image.
        """

        bounds = get_window_bounds(
            df=(
                self.X_w.pipe(pl.from_pandas).filter(
                    pl.col(mw_defs.W_TYPE) == mw_defs.LABEL_PEAK
                )
            )
        ).to_pandas()

        span_dict = {}

        import colorcet as cc
        import seaborn as sns

        
        
        bounds["color"] = sns.color_palette(cc.glasbey_dark, n_colors=bounds.shape[0])

        for i, x in bounds.groupby(mw_defs.W_IDX):

            span = hv.VSpan(
                x1=x.at[i, mw_defs.KEY_START], x2=x.at[i, mw_defs.KEY_END]
            ).opts(apply_ranges=True, color=x.at[i, "color"])

            span_dict[i] = span

        spans = hv.HoloMap(span_dict)

        return spans

    # def draw_peak_windows(
    #     self,
    # ) -> Self:
    #     draw_peak_windows(
    #         X_w=pl.from_pandas(self.X_w),
    #         w_type_key=self.w_type_key,
    #         w_idx_key=self.w_idx_key,
    #         X_idx_key=self.X_idx_key,
    #         ax=self.ax,
    #     )
    #     return self


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

    colors = sns.color_palette("deep", n_colors=len(w_idx_unique))

    peak_window_bounds_colored = peak_window_bounds.with_columns(
        color=pl.Series(colors)
    )

    grpby_obj = peak_window_bounds_colored.group_by([w_idx_key], maintain_order=True)

    for label, grp in grpby_obj:
        x0 = grp.item(0, "start")
        x1 = grp.item(0, "end")

        # testing axvspan

        ax.axvspan(
            x0,
            x1,
            label=f"peak window {label}",
            color=grp.item(0, "color"),
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
