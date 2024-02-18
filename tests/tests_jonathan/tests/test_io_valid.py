import pytest
import pandas as pd
from hplc_py.io_validation import IOValid
from typing import Any

class TestDFrameChecks:
    @pytest.fixture
    def iov(
        self,
    ) -> IOValid:
        iov = IOValid()

        return iov

    @pytest.fixture
    def empty_df(self):
        df = pd.DataFrame()
        return df

    @pytest.fixture
    def not_df(self):
        x = 0

        return x

    def test_check_df_not_df(
        self,
        iov: IOValid,
        not_df: Any,
    ) -> None:
        try:
            iov.check_df_is_pd_not_empty(not_df)
        except TypeError:
            pass

    def test_check_df_empty(
        self,
        iov: IOValid,
        empty_df: pd.DataFrame,
    ) -> None:
        try:
            iov.check_df_is_pd_not_empty(empty_df)
        except ValueError:
            pass
