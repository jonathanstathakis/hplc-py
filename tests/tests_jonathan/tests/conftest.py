from typing import Tuple, Any

import pandas as pd
import pytest
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from pandera.typing import DataFrame, Series
from hplc_py.common.common_schemas import X_Schema
from hplc_py.common.common import compute_timestep, prepare_signal_store


@pytest.fixture
def ringland_dset():
    path = "tests/tests_jonathan/test_data/a0301_2021_chris_ringland_shiraz.csv"

    dset = pd.read_csv(path)
    dset = dset[["time", "signal"]]

    return dset


@pytest.fixture
def asschrom_dset():
    path = "tests/test_data/test_assessment_chrom.csv"
    dset = pd.read_csv(path)
    return dset


@pytest.fixture
def timestep(time: Series[float]) -> float:
    timestep = compute_timestep(time)
    return timestep


@pytest.fixture
def prom() -> float:
    return 0.01


@pytest.fixture
def key_time_asschrom() -> str:
    return "x"


@pytest.fixture
def key_amp_asschrom() -> str:
    return "y"


@pytest.fixture
def prepared_data(
    asschrom_dset: pd.DataFrame,
    key_time_asschrom: str,
    key_amp_asschrom: str,
) -> tuple:
    """
    Return data as expected by the modules, containing the X_idx and X - the amplitude of the signal.
    """
    prepared_data: tuple[
        DataFrame[X_Schema],
        ndarray[Any, dtype[floating[_64Bit]]],
        ndarray[Any, dtype[floating[_64Bit]]],
    ] = prepare_signal_store(
        data=asschrom_dset, key_time=key_time_asschrom, key_amp=key_amp_asschrom
    )

    return prepared_data


@pytest.fixture
def X_data_raw(prepared_data) -> DataFrame[X_Schema]:
    """
    The raw input, simply relabeled and arranged as per `X_Schema` via `prepare_data
    """
    return prepared_data[0]


def time_array(prepared_data):
    return prepared_data[1]


def raw_array(prepared_data):
    return prepared_data[2]
