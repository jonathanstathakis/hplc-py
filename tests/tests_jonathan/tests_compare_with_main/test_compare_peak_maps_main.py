"""
2024-02-06 08:01:15 - This module is missing an actual comparison betweenthe peak mappings.
"""

import polars as pl
def test_compare_main_input_amp(
    main_peak_widths_amp_input, asschrom_amp_bcorr, main_chm_asschrom_fitted_pk
):
    df = pl.DataFrame({"main": main_peak_widths_amp_input, "mine": asschrom_amp_bcorr})

    df = df.with_columns(
        diff=(pl.col("main") - pl.col("mine")).abs().round(9), difftol=pl.lit(5)
    )

    df = df.with_columns(
        diff_prc=pl.when(pl.col("diff").ne(0))
        .then(pl.col("diff").truediv(pl.col("mine").abs().mul(100)))
        .otherwise(0)
    )

    df = df.with_columns(diffpass=pl.col("diff") < pl.col("difftol"))

    import polars.testing as pt

    left = df.select(pl.col("diffpass").all())
    right = pl.DataFrame({"diffpass": True})

    pt.assert_frame_equal(left, right)

def test_peak_map_compare(peak_map, main_pm_, timestep: float):
    """
    Compare my peak map to the main module version of the peak map.
    """
    main_peak_map_ = pl.from_pandas(main_pm_)
    peak_map = pl.from_pandas(peak_map)

    # cast the 'idx' type columns to int64 and suffix with "idx"

    idx_unit_cols = ps.matches("^*._left$|^*._right$")
    main_peak_map_ = (
        main_peak_map_.with_columns(
            idx_unit_cols.cast(pl.Int64).name.suffix("_idx")
        )
        .with_columns(idx_unit_cols.mul(timestep).name.suffix("_time"))
        .drop(idx_unit_cols)
        .with_columns(time=pl.col("t_idx") * timestep)
    )

    peak_map = peak_map.drop(
        [
            "prom",
            "prom_right",
            "prom_left",
        ]
    )

    peak_map = peak_map.melt(id_vars=["p_idx"], value_name='mine',)

    main_peak_map = main_peak_map_.melt(id_vars=["p_idx"], value_name='main')

    # inspect to see why the join might fail

    aj_my_to_main = peak_map.join(
        main_peak_map, how="anti", on=["p_idx", "variable"]
    )
    aj_main_to_my = main_peak_map.join(
        peak_map, how="anti", on=["p_idx", "variable"]
    )

    if not aj_main_to_my.is_empty() or not aj_my_to_main.is_empty():
        raise ValueError(
            f"not all keys are paired.\n\nIn my table, the following are not present in the right:{aj_my_to_main}\n\nIn main table, the following are not present in my table:{aj_main_to_my}"
        )

    df = peak_map.join(main_peak_map, on=["p_idx", "variable"], how="left")

    df = (
        df
        .with_columns(
            diff=(pl.col("mine") - pl.col("main")).abs(),
            tol_perc=pl.lit(0.05),
            av_hz=pl.sum_horizontal("mine", "main") / 2,
        )
        .with_columns(
            tol_limit=pl.col("av_hz") * pl.col("tol_perc"),
        )
        .with_columns(
            tolpass=pl.col("diff") <= pl.col("tol_limit"),
        )
    )

    assert df.filter(pl.col('tolpass')==False).is_empty()  # noqa: E712