import pandas as pd
import polars as pl
from matplotlib.axes import Axes


def plot_signal(
    df: pd.DataFrame | pl.DataFrame,
    x_colname: str,
    y_colname: str,
    label: str,
    ax: Axes,
    line_kwargs: dict = {},
) -> None:

    sig_x = df[x_colname]
    sig_y = df[y_colname]

    line = ax.plot(sig_x, sig_y, label=label, **line_kwargs)
    ax.legend(handles=line)