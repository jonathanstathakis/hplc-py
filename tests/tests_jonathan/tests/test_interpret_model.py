import pandas as pd
import pytest
from pandera.typing.pandas import DataFrame


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