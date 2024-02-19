from pandera.typing import Series
import pytest
import polars as pl

from hplc_py.chromatogram import Chromatogram
from numpy.typing import NDArray

import pandas as pd


@pytest.fixture
def __main_bcorr_interms_extract(
    main_chm_asschrom_fitted_pk,
):

    interms = main_chm_asschrom_fitted_pk._jono_bcorr_interms

    interm_signals = pl.DataFrame(
        {k: v for k, v in interms.items() if k not in ["shift", "n_iter"]}
    )
    interm_params = pl.DataFrame(
        {k: v for k, v in interms.items() if k in ["shift", "n_iter"]}
    )

    return interm_signals, interm_params


@pytest.fixture
def main_bcorr_interm_signals(__main_bcorr_interms_extract):
    """
    Contains the intermediate calculations of the main baseline correctino for asschrom.

    columns: ['signal','tfrom_new','inv_tform','bcorr_not_rounded','bcorr_rounded']
    """
    return __main_bcorr_interms_extract[0]


@pytest.fixture
def main_bcorr_interm_params(__main_bcorr_interms_extract):
    """
    contains the intermediate parameters calculated during main baseline correction of asschrom.

    dict keys: ['shift','n_iter']
    """
    return __main_bcorr_interms_extract[1]


def test_baseline_compare_main(
    asschrom_amp_bcorr,
    main_window_df,
):
    """
    Compare the differences in baseline correction between the main and my approach
    """
    import polars as pl
    from holoviews.plotting import bokeh

    bokeh.ElementPlot.width = 10000
    bokeh.ElementPlot.height = 10000

    df = (
        pl.DataFrame(
            {
                "main": main_window_df["signal_corrected"],
                "mine": asschrom_amp_bcorr,
                "amp_raw": main_window_df["signal"],
            }
        )
        .with_columns(mine=pl.col("mine"))
        .with_columns(
            main_my_diff=(pl.col("main") - pl.col("mine")).abs(),
            diff_tol=pl.lit(0.05),
        )
        .with_columns(
            main_my_diff_prc=pl.when(pl.col("main_my_diff").ne(0))
            .then(pl.col("main_my_diff").truediv(pl.col("mine").abs()))
            .otherwise(0)
        )
        .with_columns(diffpass=pl.col("main_my_diff") < pl.col("diff_tol"))
    )
    assert df.filter("diffpass" == False).is_empty()


def test_main_interms(main_bcorr_interm_params, main_bcorr_interm_signals):
    pass
