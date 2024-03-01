import distinctipy
import pandas as pd
from numpy._typing import NDArray
from pandera.typing import DataFrame
import polars as pl
from pandera.typing import Series
import numpy as np

from hplc_py.common import definitions as com_defs, common_schemas as com_schs
from hplc_py.common.common_schemas import RawData


def set_peak_color_table(
    idx: pl.DataFrame,
) -> pl.DataFrame:
    """
    Set a color table indexed by the passed idx - can be 1+ columns, thus assigning a color for each group.

    Will find the unique rows in the passed idx frame and use them to assign unique colors. These are intended to be join keys for downstream color assignment.

    Returns the indexed color table.
    """
    # find unique rows in the idx

    idx_u = idx.unique()
    colors = distinctipy.get_colors(len(idx_u))

    colors = idx_u.with_columns(pl.Series("color", colors))

    return colors


def prepare_dataset_for_input(
    data: pd.DataFrame,
    time_col: str,
    amp_col: str,
) -> DataFrame[RawData]:
    """
    Rename the x and y columns to match SignalDFLoaded schema.

    :param data: the dirty dataset
    :type data: pd.DataFrame
    :param time_col: the x column to be renamed to match the schema
    :type time_col: str
    :param amp_col: the y column to be renamed to match the schema
    :type time_col: str
    """
    data = data.rename(
        {time_col: RawData.time, amp_col: RawData.amp}, axis=1, errors="raise"
    ).reset_index(names="t_idx")

    data = DataFrame[RawData](data)

    return data


def compute_timestep(time_array: Series[float]) -> float:
    # Define the average timestep in the chromatogram. This computes a mean
    # but values will typically be identical.

    dt = np.diff(time_array)
    mean_dt = np.mean(dt)
    return mean_dt.astype(float)


def prepare_signal_store(
    data: pd.DataFrame,
    key_time: str = com_defs.KEY_TIME,
    key_amp: str = com_defs.KEY_AMP,
) -> pl.DataFrame:
    """
    Relabel data to match definitions. In this pipeline, the amplitude is defined as X and time values are disregarded beyond an observation of the sampling rate. Return X_data as pandas dataframe as per X_Schema, and time as numpy array.
    """

    # select the designated time and X columns and rename as per internal key values and add an index col as int64

    tbl_signal: pl.DataFrame = (
        data.pipe(pl.from_pandas)
        .with_row_index(name=com_defs.X_IDX)
        .select(
            [
                pl.col(com_defs.X_IDX).cast(int),
                pl.col(key_time).alias(com_defs.X),
                pl.col(key_amp).alias(com_defs.KEY_TIME),
            ]
        )
    )
    
    return tbl_signal
