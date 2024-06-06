from dataclasses import dataclass
from findiff import FinDiff
from hplc_py.definitions import PRECISION
from hplc_py.datasets import DataSets
from hplc_py.pipeline.preprocess_dashboard import PreProcesser
from pandera.typing.polars import DataFrame, LazyFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

import holoviews as hv
import hvplot
import numpy as np
import pandera as pa
import panel as pn
import polars as pl

pl.Config.set_fmt_float(fmt="full")
pl.Config.set_tbl_rows(10)

hv.renderer("bokeh").theme = "dark_minimal"

dsets = DataSets()
ringland = dsets.ringland.fetch().with_columns(
    pl.col("time").round(9), pl.col("signal").round(9)
)
