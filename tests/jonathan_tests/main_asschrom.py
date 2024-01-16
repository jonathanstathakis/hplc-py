import pytest
import hplc
from pandera.typing import DataFrame

@pytest.fixture
def main_chm_cls(
    in_signal: DataFrame,
    ):
        main_chm = hplc.quant.Chromatogram(in_signal)

        return main_chm