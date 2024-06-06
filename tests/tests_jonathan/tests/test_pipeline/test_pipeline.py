from hplc_py.pipeline.pipeline import DeconvolutionPipeline
import pytest
import pandera as pa
from pandera.typing import DataFrame
from hplc_py.common.common_schemas import RawData
import pandas as pd
import pickle
import os
import polars as pl


@pytest.fixture
def asschrom_pk_filepath():
    pk_fn = "asschrom_pipeline"
    pkpth = os.path.join(
        os.getcwd(), "tests", "tests_jonathan", "tests", "test_pipeline", pk_fn + ".pk"
    )
    return pkpth


@pytest.fixture
def asschrom_pipeline(
    asschrom_dset: pd.DataFrame,
    asschrom_pk_filepath,
):
    """
    pickle the pipeline object in this dir location
    """
    if not os.path.isfile(asschrom_pk_filepath):

        pipeline = DeconvolutionPipeline()

        pipeline.run(
            data=asschrom_dset,
            key_time="x",
            key_amp="y",
        )

        with open(asschrom_pk_filepath, "wb") as f:
            pickle.dump(pipeline, f)

    with open(asschrom_pk_filepath, "rb") as f:
        pipeline = pickle.load(f)

    return pipeline


def test_pipeline(asschrom_dset: pd.DataFrame, asschrom_pipeline):
    pipeline = DeconvolutionPipeline()

    pipeline.run(
        data=asschrom_dset,
        key_time="x",
        key_amp="y",
    )
    from polars.testing import assert_frame_equal

    assert_frame_equal(
        pipeline.tbl_fit_report,
        asschrom_pipeline.tbl_fit_report,
    )

    assert True

    # TODO:
    # - [x] pickle pipeline
    # - [ ] define a eq method to compare the attributes of pipeline for equality
