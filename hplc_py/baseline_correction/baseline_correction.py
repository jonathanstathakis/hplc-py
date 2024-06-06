import deprecation
import polars as pl
import pandas as pd
from hplc_py.baseline_correction.compute_background import ComputeBackground
from hplc_py.baseline_correction import definitions as bc_defs
from hplc_py.baseline_correction.viz import VizBCorr
from hplc_py.common import definitions as com_defs
from hplc_py.reports import report
from hplc_py.transformer_abc import transformer_abc
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Self
from hplc_py.baseline_correction import compute_background
from hplc_py import precision
import holoviews as hv

from hplc_py import transformers


class SNIPBCorr(
    TransformerMixin,
    BaseEstimator,
    precision.Precision,
    transformers.TransformerPolarsAdapter,
):

    def __init__(self, n_iter=5, verbose: bool = True):
        self.n_iter = n_iter
        self.verbose = verbose

    def fit(self, X, y=None) -> Self:
        self._X = X
        self.feature_names_in_ = X.columns

        return self

    def transform(self, X, y=None):
        """
        TODO: modify `compute_background` to work with 2D arrays
        """

        # scikit_learn works in 2D as the minimum dimensions, but `compute_background` is currently set up for 1d arrays

        if not isinstance(X, pl.DataFrame):
            raise TypeError("Please input a polars DataFrame")

        if len(X.columns) > 1:
            raise ValueError("Please provide a 1 column polars DataFrame")

        column_name = X.columns[0]

        self._X = X
        self.feature_names_in_ = X.columns

        background = compute_background.compute_background(
            amp=X.select(column_name).to_series().to_numpy().ravel(),
            n_iter=self.n_iter,
            verbose=self.verbose,
        )

        self._background_ = pl.Series(name="background", values=background)
        # ColumnTransformer expects 2D output, thus reshape the output of compute_background to 2D

        self._X_ = self.X.select(pl.col(column_name).sub(self.background_))

        return self.X_

    def get_feature_names_out(self):
        return self.feature_names_in_

    def plot(self):

        return self.data_as_frame.plot(
            y=["X", "X_", "background"],
            alpha=[0.8, 0.8, 0.8],
            line_dash=["solid", "solid", "dashdot"],
            title="SNIP background subtraction",
        )

    @property
    def signals_(self):
        raise AttributeError(
            ".signal_ has been removed, state management should have been done externally. Fix it now."
        )

    @property
    def X(self):
        return self._X

    @X.getter
    def X(self):
        return self._X.with_columns(pl.all().round(self._precision))

    @property
    def background_(self):
        return self._background_

    @background_.getter
    def background_(self):
        return self._background_.round(self._precision)

    @property
    def X_(self):
        return self._X_

    @X_.getter
    def X_(self):
        return self._X_.with_columns(pl.all().round(self._precision))

    @property
    def data_as_frame(self):
        return (
            pl.DataFrame(
                dict(
                    X=self.X.select(self.feature_names_in_[0]),
                    X_=self.X_.select(self.feature_names_in_[0]),
                    background=self.background_.to_frame(),
                )
            )
            .with_row_index("idx")
            .cast({"idx": int})
        )

    def plot_by_quantiles(
        self,
        n_quantiles: int = 4,
        plot_opts: dict = {},
    ):
        """
        plot the input, background and corrected signals as overlaid subplots divided
        into "n" subplots according to `n_quantiles`, defaulting to 4.

        Allows for finer grained observation of the signal across the time axis and
        comparison of different regions.

        n_quantiles: number of quantiles by which to divide the time axis, corresponding
        to the number of subplots produced.

        plot_opts: a dict of options for the individual plots. Although hvplot provides
        the .`opts` method on plot objects, this function returns a NdLayout, masking the
        subplot opts. Easiest way to configure the plot itself is on generation of the
        mapping. Refer to hvplot.Line ('bokeh') for args.
        """

        quantiled_data = self.get_quantiled_data(n_quantiles=n_quantiles)

        plots = {
            k: df.plot(
                x="idx",
                y=["X", "X_", "background"],
                line_alpha=[0.8, 0.9, 0.8],
                line_dash=["dashdot", "solid", "dotted"],
                grid=True,
            ).opts(**plot_opts)
            for k, df in quantiled_data.partition_by(
                maintain_order=True, by=["quantile_rank"], as_dict=True
            ).items()
        }

        layout = (
            hv.NdLayout(plots, kdims="quartile")
            # .cols(2)
            .opts(
                shared_axes=False,
                title=f"baseline correction",
            )
        )

        return layout


@deprecation.deprecated
class BaselineCorrection(transformer_abc.Transformer):
    """
    A 'batteries included' baseline correction module which takes a 1D signal and computes the background, subtracting it and returning a dataframe of time index, raw signal, background, and corrected signal.

    It includes a viz method via hvplot which will produce a plot object that displays an overlay of the raw, corrected and background curves. To use, call `hvplot.show(plot_obj)`
    """

    def __init__(
        self,
        n_iter: int,
        verbose: bool,
    ):
        self.compute_background = ComputeBackground(n_iter=n_iter, verbose=verbose)

        self.vizualiser = VizBCorr()

        self._report = report.Report(transformer_name="baseline_correction")

    def fit(
        self,
        X,
    ):
        self._X = X
        self.compute_background.fit(X=X)

    def transform(
        self,
    ) -> pd.DataFrame:
        if self._X.size == 0:
            raise AttributeError(
                "internal X array is empty, probably need to call `fit` first"
            )

        background = self.compute_background.transform()

        self.corrected = self._X - background

        self.set_report()

        return self.corrected

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(
            X,
        )
        X_t = self.transform()

        return X_t

    def set_report(self):
        """
        Collect the input, background and corrected signals together in a long form frame to be grouped by 'signal' for plotting an overlay
        """
        signals = (
            pl.DataFrame(
                {
                    "input": self._X,
                    "background": self.compute_background.background,
                    "corrected": self.corrected,
                }
            )
            .with_row_index("idx")
            .melt(id_vars="idx", value_name="amp", variable_name="signal")
            .to_pandas()
        )

        bcorr_plot_obj = self.vizualiser.viz_baseline_correction(signals=signals)

        self._report.tables = signals

        self._report.viz = bcorr_plot_obj

    @property
    def report(self):
        return self._report
