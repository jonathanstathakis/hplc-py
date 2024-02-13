import matplotlib as mpl
from hplc_py.hplc_py_typing.hplc_py_typing import Data
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pandera as pa
import polars as pl
from matplotlib.axes import Axes as Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from numpy import float64
from pandera.typing import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import (
    WindowedSignal,
)

from hplc_py.io_validation import IOValid

"""
Module for vizualisation, primarily the "Show" class
"""


    
class Show(
):
    def __init__(self):
        pass

    def plot_signal(
        self,
        signal_df: pd.DataFrame,
        time_col: str,
        amp_col: str,
        ax: plt.Axes,
    ):
        x = signal_df[time_col]
        y = signal_df[amp_col]

        ax.plot(x, y, label="bc chromatogram")
        return ax

    def plot_reconstructed_signal(
        self,
        unmixed_df,
        ax,
    ):
        """
        Plot the reconstructed signal as the sum of the deconvolved peak series
        """
        amp_unmixed = (
            unmixed_df.pivot_table(columns="p_idx", values="amp_unmixed", index="time")
            .sum(axis=1)
            .reset_index()
            .rename({0: "amp_unmixed"}, axis=1)
        )
        x = amp_unmixed["time"]
        y = amp_unmixed["amp_unmixed"]

        ax.plot(x, y, label="reconstructed signal")

        return ax
