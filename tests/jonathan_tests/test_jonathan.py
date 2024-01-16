"""
Best Practices:
    - Define Fixture classes which are inherited into Test classes.
    - Define stateless method classes which provide methods for a master state storage class.

"""
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure
from numpy import floating
from numpy.typing import NDArray
from pandera.typing.pandas import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import (
    OutParamsBase,
    Popt,
    Recon,
    SignalDF,
)
from hplc_py.hplc_py_typing.interpret_model import interpret_model
from hplc_py.map_signals.map_peaks import PeakMap
from hplc_py.quant import Chromatogram

from .conftest import AssChromResults

OutPeakDFAssChrom = PeakMap


pd.options.display.precision = 9
pd.options.display.max_columns = 50


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


manypeakspath = "/Users/jonathan/hplc-py/tests/test_data/test_many_peaks.csv"
asschrompath = "tests/test_data/test_assessment_chrom.csv"


def test_acr(acr: AssChromResults):
    assert acr
    pass


def test_param_df_adapter(acr: AssChromResults):
    param_df = acr.tables["asschrom_param_tbl"]

    adapted_param_df = acr.adapt_param_df(param_df)
    try:
        OutParamsBase(adapted_param_df)
    except Exception as e:
        raise RuntimeError(e)


def check_df_exists(df):
    assert isinstance(df, pd.DataFrame)
    assert df.all
    return None


def test_target_window_df_exists(target_window_df: Any):
    check_df_exists(target_window_df)
    pass


def test_get_asschrom_results(acr: AssChromResults):
    """
    Simply test whether the member objects of AssChromResults are initialized.
    """

    for tbl in acr.tables:
        check_df_exists(acr.tables[tbl])


def test_timestep_exists_and_greater_than_zero(timestep: float):
    assert timestep
    assert timestep > 0


def test_amp_raw_not_null(amp_raw):
    """
    for exploring shape and behavior of amp. a sandpit
    """
    assert all(amp_raw)


class TestTimeStep:
    def test_timestep(self, timestep: float):
        assert timestep


class TestLoadData:
    def test_loaded_signal_df(
        self,
        loaded_signal_df: DataFrame[SignalDF],
    ):

        SignalDF(loaded_signal_df, lazy=True)


def test_popt_to_parquet(
    popt_df: Any,
    popt_parqpath: Literal["/Users/jonathan/hplc-py/tests/jonathan_tests/assch…"],
):
    """
    A function used to produce a parquet file of a popt df. I suppose it itself acts as a test, and means that whenever i run the full suite the file will be refreshed.
    """

    popt_df.to_parquet(popt_parqpath)


class TestShow:
    """
    Test the Show class methods
    """

    @pytest.fixture
    def fig_ax(self):
        return plt.subplots(1)

    @pytest.fixture
    def decon_out(
        self,
        chm: Chromatogram,
        signal_df: DataFrame,
        peak_df: DataFrame[PeakMap],
        window_df: DataFrame[Any],
        timestep: float,
    ):
        return chm._deconvolve.deconvolve_peaks(signal_df, peak_df, window_df, timestep)

    @pytest.fixture
    def popt_df(
        self, decon_out: tuple[DataFrame[Popt], DataFrame[Recon]]
    ):
        return decon_out[0]

    @pytest.fixture
    def popt_df(
        self, decon_out: tuple[DataFrame[Popt], DataFrame[Recon]]
    ):
        return decon_out[0]

    def test_plot_raw_chromatogram(
        self,
        fig_ax: tuple[Figure, Any],
        chm: Chromatogram,
        signal_df: SignalDF,
    ):
        chm._show.plot_raw_chromatogram(
            signal_df,
            fig_ax[1],
        )

    def test_plot_reconstructed_signal(
        self,
        chm: Chromatogram,
        fig_ax: tuple[Figure, Any],
        unmixed_df: DataFrame[Recon],
    ):
        chm._show.plot_reconstructed_signal(unmixed_df, fig_ax[1])

    def test_plot_individual_peaks(
        self,
        chm: Chromatogram,
        fig_ax: tuple[Figure, Any],
        unmixed_df: DataFrame[Recon],
    ):
        ax = fig_ax[1]

        chm._show.plot_individual_peaks(
            unmixed_df,
            ax,
        )

    def test_plot_overlay(
        self,
        chm: Chromatogram,
        fig_ax: tuple[Figure, Any],
        signal_df: DataFrame,
        unmixed_df: DataFrame[Recon],
    ):
        fig = fig_ax[0]
        ax = fig_ax[1]
        chm._show.plot_raw_chromatogram(
            signal_df,
            ax,
        )
        chm._show.plot_reconstructed_signal(
            unmixed_df,
            ax,
        )
        chm._show.plot_individual_peaks(
            unmixed_df,
            ax,
        )
