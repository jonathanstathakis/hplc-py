import polars as pl
from pandera.typing import DataFrame
from hplc_py.hplc_py_typing.hplc_py_typing import PSignals
import polars.selectors as ps
import pytest
from typing import Any


@pytest.fixture
def main_psignals(
    main_chm_asschrom_fitted_pk: Any,
):
    """
    The main package asschrom unmixed peak signals.

    Presented in long format in columns: ['t_idx','p_idx','main']
    """
    main_psignals = (
        pl.DataFrame(
            main_chm_asschrom_fitted_pk.unmixed_chromatograms,
            schema={
                str(p): pl.Float64
                for p in range(
                    main_chm_asschrom_fitted_pk.unmixed_chromatograms.shape[1]
                )
            },
        )
        .with_row_index("t_idx")
        .melt(id_vars="t_idx", variable_name="p_idx", value_name="main")
        .with_columns(pl.col("p_idx").cast(pl.Int64))
    )

    return main_psignals


def test_p_signals_compare_main(
    main_psignals: pl.DataFrame,
    psignals: DataFrame[PSignals],
):
    """
    Test the asschrom dataset through my pipeline compared to main. Takes the output
    individual peak signals and compares the amplitude values of each peak to main.
    Total variation greater than 5% is treated as a failure.
    
    Note: This has untested behavior in the case that the number of peaks differ.
    """
    psignals_: pl.DataFrame = (
        pl.from_pandas(psignals).drop("time").rename({"amp_unmixed": "mine"})
    )

    signals = (
        (
            psignals_.with_columns(pl.col("t_idx").cast(pl.UInt32))
            .join(
                main_psignals,
                how="left",
                on=["p_idx", "t_idx"],
            )
            .select(["p_idx", "t_idx", "mine", "main"])
            .melt(
                id_vars=["p_idx", "t_idx"],
                value_vars=["mine", "main"],
                value_name="amp_unmixed",
                variable_name="source",
            )
            .with_columns(ps.float().round(8))
            .pivot(index=["p_idx", "t_idx"], columns="source", values="amp_unmixed")
            .with_columns(
                hz_av=pl.sum_horizontal("mine", "main") / 2,
                tol_perc=0.05,
                diff=(pl.col("mine") - pl.col("main")),
            )
            .with_columns(
                tol_act=pl.col("hz_av") * pl.col("tol_perc"),
            )
            .with_columns(
                tol_pass=pl.col("diff") <= pl.col("tol_act"),
            )
        )
        .group_by("p_idx")
        .agg(
            perc_True=pl.col("tol_pass")
            .filter(tol_pass=True)
            .count()
            .truediv(pl.col("tol_pass").count())
            .mul(100),
            perc_False=pl.col("tol_pass")
            .filter(tol_pass=False)
            .count()
            .truediv(pl.col("tol_pass").count())
            .mul(100),
        )
        .sort("p_idx")
    )

    import polars.testing as pt

    pt.assert_frame_equal(
        signals.select((pl.col("perc_False") < 5).all()),
        pl.DataFrame({"perc_False": True}),
    )
