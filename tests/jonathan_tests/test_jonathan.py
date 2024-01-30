"""
Best Practices:
    - Define Fixture classes which are inherited into Test classes.
    - Define stateless method classes which provide methods for a master state storage class.

"""
from hplc_py.show import PlotSignal
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure
from numpy import floating
from numpy.typing import NDArray
from pandera.typing.pandas import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import (
    Params,
    Popt,
    PSignals,
    SignalDFLoaded,
)
from hplc_py.hplc_py_typing.interpret_model import interpret_model
from hplc_py.map_signals.map_peaks.map_peaks import PeakMap

OutPeakDFAssChrom = PeakMap


pd.options.display.precision = 9
pd.options.display.max_columns = 50

import pandera as ap


@pytest.mark.skip(reason="Currently not in use, not obvious error occuring from schema_str")
class TestInterpretModel:
    def schema_cls(
        self,
        schema_str,
    ):
        # instantiate the schema class
        exec(schema_str)

        # get the schema class object from locals
        schema_cls = locals()["InSampleDF"]

        return schema_cls

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [-1, -2, -3, -4, -5],
            }
        )

    @pytest.fixture
    def test_gen_sample_df_schema(self, sample_df: DataFrame):
        interpret_model(sample_df)

        return None

    @pytest.fixture
    def eq_schema_str(
        self,
        sample_df: DataFrame,
    ):
        check_dict = {col: "eq" for col in sample_df.columns}
        schema_def_str = interpret_model(sample_df, "InSampleDF", "", check_dict=check_dict)

        return schema_def_str

    @pytest.fixture
    def isin_schema_str(
        self,
        sample_df: DataFrame,
    ):
        check_dict = {col: "isin" for col in sample_df.columns}
        schema_def_str = interpret_model(sample_df, "InSampleDF", "", check_dict=check_dict)

        return schema_def_str

    @pytest.fixture
    def basic_stats_schema_str(self, sample_df: DataFrame):
        numeric_cols = []
        non_numeric_cols = []
        for col in sample_df:
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)

        # test numeric cols with 'basic_stats'
        numeric_col_check_dict = dict(
            zip(numeric_cols, ["basic_stats"] * len(numeric_cols))
        )
        non_numeric_col_check_dict = dict(
            zip(non_numeric_cols, ["isin"] * len(non_numeric_cols))
        )

        check_dict = dict(**numeric_col_check_dict, **non_numeric_col_check_dict)
        schema_def_str = interpret_model(sample_df, "InSampleDF", "", check_dict=check_dict)

        return schema_def_str

    def test_eq_schema(
        self,
        sample_df: DataFrame,
        eq_schema_str: str,
    ):
        """
        Test whether the 'equals' schema works as expected
        """
        schema = self.schema_cls(eq_schema_str)
        schema(sample_df)

    def test_isin_schema(
        self,
        sample_df: DataFrame,
        isin_schema_str: str,
    ):
        """
        Test whether the 'isin' schema works as expected
        """
        schema = self.schema_cls(isin_schema_str)
        schema(sample_df)

    def test_basicstats_schema(
        self,
        sample_df: DataFrame,
        basic_stats_schema_str: str,
    ):
        """
        Test whether the 'basic_stats' schema works as expected
        """

        schema = self.schema_cls(basic_stats_schema_str)
        schema(sample_df)

    def test_base_schema(self, sample_df: DataFrame):
        # schema = self.schema_cls(sample_df)
        interpret_model(sample_df, "TestBaseSchema", is_base=True)


def schema_error_str(schema, e, df):
    err_str = "ERROR REPORT:"
    err_str += "ERROR: " + str(e) + "\n"
    err_str += "SCHEMA: " + str(schema) + "\n"
    err_str += "ACTUALS:\n"
    for col in df:
        err_str += str(col) + "\n"
        err_str += str(df[col].tolist()) + "\n"

    err_str += "compare these against the schema and replace where necessary"
    raise RuntimeError(err_str)


def schema_error_str_long_frame(schema, e, df):
    err_str = "ERROR REPORT:"
    err_str += "ERROR: " + str(e) + "\n"
    err_str += "SCHEMA: " + str(schema) + "\n"
    err_str += "ACTUALS:\n"
    for col in df:
        err_str += str(col) + "\n"
        col_min = df[col].min()
        col_max = df[col].max()
        err_str += f"{{'min_value':{col_min}, 'max_value':{col_max}}}" + "\n"

    err_str += "compare these against the schema and replace where necessary"
    raise RuntimeError(err_str)

