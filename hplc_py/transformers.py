from deprecation import deprecated
import numpy.typing as npt
from scipy import signal
from hplc_py import definitions
from sklearn.base import BaseEstimator, TransformerMixin
from findiff import FinDiff
import holoviews as hv
import hvplot
import numpy as np
import pandera as pa
import panel as pn
import polars as pl
import holoviews as hv


class TransformerPolarsAdapter:
    _input_key = "input"
    _output_key = "output"
    feature_names_in_ = []
    X = None
    X_ = None

    @property
    def X_as_pl(self):
        return (
            pl.Series(name=self.feature_names_in_[0], values=self.X)
            .rename(self._input_key)
            .to_frame()
        )

    @property
    def X__as_pl(self):
        return (
            pl.Series(name=self.feature_names_in_[0], values=self.X_)
            .rename(self._output_key)
            .to_frame()
        )

    def get_quantiled_data(self, n_quantiles: int):

        quantiled_data = self.data_as_frame.with_columns(
            pl.col("idx").qcut(quantiles=n_quantiles).alias("quantiles")
        ).with_columns(
            pl.col("idx").first().over("quantiles").rank("dense").alias("quantile_rank")
        )

        return quantiled_data

    @property
    def data_as_frame(self):
        return pl.concat([self.X_as_pl, self.X__as_pl], how="horizontal")

    def plot(self, opts: dict = {}):
        data = self.data_as_frame

        plot_obj = data.plot(
            y=[self._input_key, self._output_key], line_alpha=[0.8, 0.8]
        ).opts(**opts)

        return plot_obj


class ScondOrderFiniteDiffSub(
    BaseEstimator, TransformerMixin, TransformerPolarsAdapter
):
    """
    wrapper around FiniteDiffAdder to provide subtraction of the second order finite difference
    """

    def __init__(self, k: float = 1):
        order = 2
        dx = 1
        sign = "-"
        self.k = k

        self._finite_diff_adder = FiniteDiffAdder(
            order=order, dx=dx, sign=sign, k=self.k
        )

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]

        self.is_fitted_ = True

        self.feature_names_in_ = X.columns

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "n_features_")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                "shape of input is different from what was seen" "in `fit`"
            )

        self.X = X
        self.feature_names_in_ = X.columns

        self.X_ = self._finite_diff_adder.fit_transform(X=self.X)

        return self.X_

    def get_feature_names_out(self):
        """
        dummy method to enable instantiation of `set_output` API, see: <https://stackoverflow.com/a/75036830/21058408>
        """
        return self.feature_names_in_

    def plot(self):
        return self._finite_diff_adder.plot()


class FirstOrderFiniteDiffAddition(
    BaseEstimator, TransformerMixin, TransformerPolarsAdapter
):
    """
    wrapper around FiniteDiffAdder to provide addition of the first order finite difference
    """

    def __init__(self, k: float = 1):
        order = 1
        dx = 1
        sign = "+"
        self.k = k

        self._finite_diff_adder = FiniteDiffAdder(
            order=order, dx=dx, sign=sign, k=self.k
        )

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]

        self.is_fitted_ = True

        self.feature_names_in_ = X.columns

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "n_features_")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                "shape of input is different from what was seen" "in `fit`"
            )

        self.X = X
        self.feature_names_in_ = X.columns

        self.X_ = self._finite_diff_adder.fit_transform(X=self.X)

        return self.X_

    def get_feature_names_out(self):
        """
        dummy method to enable instantiation of `set_output` API, see: <https://stackoverflow.com/a/75036830/21058408>
        """
        return self.feature_names_in_

    def plot(self):
        return self._finite_diff_adder.plot()

from typing import Literal

class FiniteDiffAdder(BaseEstimator, TransformerMixin, TransformerPolarsAdapter):
    """
    Use to add or subtract finite differences from X. combine multiple steps,
    as stages in the pipeline, for example subtract the second derivative, add the 4th.

    The subtraction of the second derivative helps with overlap but is only good if SNR
    is already high as it increases noise

    The addition of the first derivative helps symmetrize peaks, but moves peaks slightly left
    """

    def __init__(self, order: int = 1, dx: float = 1, k: float = 1, sign: str = "-"):
        """
        order: the order of the finite difference, 1st, 2nd, 3rd, etc.
        sign: ["+", "-"], whether to add or subtract the finite difference from X
        k: second derivative weighting term

        notes:

        use subtraction with second order, addition with first order
        """
        self.order = order
        self.dx = dx
        self.sign = sign
        self.k = k
        self.findiff_operator = FinDiff(0, dx, order)

    def fit(self, X, y=None):

        if self.sign not in ["+", "-"]:
            raise ValueError("please input '+' or '-'")

        self.n_features_ = X.shape[1]

        self.is_fitted_ = True

        self.feature_names_in_ = X.columns

        return self

    def transform(self, X, y=None):
        """
        logic here
        """

        check_is_fitted(self, "n_features_")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                "shape of input is different from what was seen" "in `fit`"
            )

        self.X = X

        assert not self.X.is_empty()

        self.feature_names_in_ = X.columns

        diff_input = X.get_column("signal").to_numpy()

        diff_array = self.findiff_operator(diff_input)
        
        self._derivative_ = diff_array
        self._derivative_weighted_ = diff_array * self.k
        if self.sign == "+":
            self.X_ = self.X.select(pl.col(self.feature_names_in_[0]).add(self._derivative_weighted_))

        elif self.sign == "-":
            self.X_ = self.X.select(pl.col(self.feature_names_in_[0]).sub(self._derivative_weighted_))

        return self.X_

    def get_feature_names_out(self):
        """
        dummy method to enable instantiation of `set_output` API, see: <https://stackoverflow.com/a/75036830/21058408>
        """
        return self.feature_names_in_

    @property
    def derivative_(self):

        return self._derivative_
    
    @property
    def data_as_frame(self):

        from dataclasses import dataclass

        @dataclass
        class Columns:
            order: str = "order"
            k: str = "k"
            X: str = "X"
            X_: str = "X_"
            diff: str = "diff"
            diff_weighted: str = "diff_weighted"

        # establish the column name data obj
        self._cols = Columns()

        if self.X_ is None:
            raise RuntimeError("Run `fit_transform` first")
        return (
            pl.DataFrame({
                self._cols.order: self.order,
                self._cols.X: self.X,
                self._cols.k: self.k,
                self._cols.diff: self.derivative_,
                self._cols.diff_weighted: self._derivative_weighted_,
                self._cols.X_:self.X_,
                }
            )
            .with_row_index("idx")
            .select(pl.col("idx"), pl.exclude("idx"))
        )

    @property
    def operator_dict(self):
        return {
            "+": "addition",
            "-": "subtraction",
        }

    @property
    def order_dict(self):
        return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}

    def plot(self):

        return self.data_as_frame.plot(
            y=["X_", "X", "diff_weighted"],
            line_alpha=[0.8, 0.8, 0.8],
            line_dash=["solid", "dotdash", "dotted"],
            title=f"{self.order_dict[self.order]} order derivative {self.operator_dict[self.sign]} with weighting = {self.k}",
        )

    def plot_by_quantiles(self, n_quantiles: int = 4, plot_opts: dict = {}):

        quantiled_data = self.get_quantiled_data(n_quantiles=n_quantiles)

        plots = {
            k: df.plot(
                x="idx",
                y=["X", "X_", "diff_weighted"],
                line_alpha=[0.8, 0.9, 0.8],
                line_dash=["dashdot", "solid", "dotted"],
                grid=True,
            ).opts(**plot_opts)
            for k, df in quantiled_data.partition_by(
                maintain_order=True, by=["quantile_rank"], as_dict=True
            ).items()
        }

        layout = hv.NdLayout(plots, kdims="quartile").opts(
            shared_axes=False,
            title=f"{self.order_dict[self.order]} order derivative {self.operator_dict[self.sign]} with weighting = {self.k}, by idx quantiles",
        )

        return layout


from sklearn.utils.validation import check_array, check_is_fitted


class SavgolFilter(BaseEstimator, TransformerMixin, TransformerPolarsAdapter):
    """
    Use to smooth a signal
    """

    def __init__(
        self,
        window_length: int,
        polyorder: int,
        deriv: int = 0,
        delta: float = 1.0,
        mode: str = "interp",
    ):
        """ """
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.mode = mode

    def fit(self, X, y=None):

        self.n_features_ = X.shape[1]

        self.is_fitted_ = True

        self.feature_names_in_ = X.columns

        return self

    def transform(self, X, y=None):
        """
        logic here
        """

        check_is_fitted(self, "n_features_")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                "shape of input is different from what was seen" "in `fit`"
            )

        self.X = X
        self.feature_names_in_ = X.columns

        input_x = self.X.to_series(0).to_numpy().ravel()

        self.X_: npt.NDArray = signal.savgol_filter(
            x=input_x,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            mode=self.mode,
        )

        return self.X_

    def get_feature_names_out(self):
        """
        dummy method to enable instantiation of `set_output` API, see: <https://stackoverflow.com/a/75036830/21058408>
        """
        return self.feature_names_in_


class PowerLaw(BaseEstimator, TransformerMixin, TransformerPolarsAdapter):
    """
    Use to smooth a signal
    """

    def __init__(
        self,
        order: int = 2,
    ):
        """
        order: the order of the finite difference, 1st, 2nd, 3rd, etc.
        sign: ["+", "-"], whether to add or subtract the finite difference from X
        k: second derivative weighting term
        """
        self.order = order

    def fit(self, X, y=None):

        self.n_features_ = X.shape[1]

        self.is_fitted_ = True

        self.feature_names_in_ = X.columns

        return self

    def transform(self, X, y=None):
        """
        logic here
        """

        check_is_fitted(self, "n_features_")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                "shape of input is different from what was seen" "in `fit`"
            )

        self.X = X
        self.feature_names_in_ = X.columns

        input_x = self.X.to_series(0).to_numpy().ravel()

        self.X_t = np.power(self.X, self.order)

        return self.X_t

    def get_feature_names_out(self):
        """
        dummy method to enable instantiation of `set_output` API, see: <https://stackoverflow.com/a/75036830/21058408>
        """
        return self.feature_names_in_


class Rounder(BaseEstimator, TransformerMixin, TransformerPolarsAdapter):
    """
    Round X to a given precision
    """

    def __init__(
        self,
        precision: int = definitions.PRECISION,
    ):
        """
        precision: the number of decimal places to round to.
        """
        self.precision = precision

    def fit(self, X, y=None):
        self.X = X
        self.n_features_ = X.shape[1]

        self.is_fitted_ = True

        self.feature_names_in_ = X.columns

        return self

    def transform(self, X, y=None):
        """
        logic here
        """

        check_is_fitted(self, "n_features_")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                "shape of input is different from what was seen" "in `fit`"
            )

        self.X = X
        self.feature_names_in_ = X.columns
        self.X_ = self.X.with_columns(pl.all().round(self.precision))
        return self.X_

    def get_feature_names_out(self):
        """
        dummy method to enable instantiation of `set_output` API, see: <https://stackoverflow.com/a/75036830/21058408>
        """
        return self.feature_names_in_


class IntermediateTransformer:
    """
    store the name, transformer object, input, output and provide plotting functionality
    """

    def __init__(self, name: str, tformer):
        assert isinstance(name, str)

        self._tformer = tformer
        self.name = name
        self._input: pl.Series = tformer.X_as_pl
        self._output: pl.Series = tformer.X__as_pl

    def plot(self):
        return self._tformer.plot()

    def __repr__(self):
        repr_dict = {
            "name": self.name,
            "input": {"type": type(self._input), "shape": self._input.shape},
            "output": {"type": type(self._output), "shape": self._output.shape},
        }
        import pprint

        return pprint.pformat(repr_dict, sort_dicts=False)


class IntermediateExtractor:
    """
    take a sklearn Pipeline object and invert it to reveal the internal transformers.

    Use to access intermediate values and interfaces present in the transformers
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._int_tformers = self._int_tformers_as_dict()

    def _get_steps(self):
        return self._pipeline.steps

    def _get_order(self):
        return list(range(len(self._get_steps()) + 1))

    def _int_tformers_as_dict(self):
        """
        return a dict of the transformers with keys as order of appearance
        """
        steps = self._get_steps()

        order = self._get_order()

        intermediate_tformers = [
            IntermediateTransformer(name=step[0], tformer=step[1]) for step in steps
        ]

        tformers = dict(zip(order, intermediate_tformers))

        return tformers

    def get_tformer_by_order(self, int):
        """
        select a given transformer by its order of appearance
        """
        return self._int_tformers[int]

    def get_tformer_names(self):
        return [tformer.name for tformer in self._int_tformers.values()]

    def get_tformer_by_name(self, name):
        """
        select a given transformer by its name. Since sklearn Pipeline requires unique names, there is no ambiguity.
        """

        if name not in self.get_tformer_names():
            raise ValueError(
                f"{name} not in tformer list.\nLegal names: {self.get_tformer_names()}"
            )
        idx = [
            idx
            for idx in self._int_tformers.keys()
            if self._int_tformers[idx].name == name
        ][0]

        return self._int_tformers[idx]

    def __repr__(self):
        import pprint

        return pprint.pformat(self._int_tformers, sort_dicts=False)
