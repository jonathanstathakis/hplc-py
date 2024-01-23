import polars as pl
pl.Config(set_tbl_cols=50)
import polars.selectors as ps
import polars.testing as polt

from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandera.typing import Series, DataFrame
import pandera as pa

from hplc_py.baseline_correct.correct_baseline import SignalDFBCorr
from hplc_py.hplc_py_typing.hplc_py_typing import (
    FloatArray,
)
from hplc_py.map_signals.checks import DFrameChecks
from hplc_py.map_signals.map_peaks import (
    WHH,
    FindPeaks,
    MapPeakPlots,
    MapPeaks,
    MapPeaksMixin,
    PeakBases,
    PeakMap,
)


class TestMapPeaksFix:
    @pytest.fixture
    def mpm(
        self,
    ) -> MapPeaksMixin:
        mpm = MapPeaksMixin()

        return mpm

    @pytest.fixture
    def mp(
        self,
    ) -> MapPeaks:
        mp = MapPeaks()
        return mp

    @pytest.fixture
    def prom(self) -> float:
        return 0.01

    @pytest.fixture
    def wlen(self) -> None:
        return None

    @pytest.fixture
    def fp(
        self,
        amp_bcorr: FloatArray,
        time: FloatArray,
        prom: float,
        mpm: MapPeaksMixin,
        wlen: None,
    ) -> DataFrame[FindPeaks]:
        fp = mpm._set_fp_df(
            amp_bcorr,
            time,
            prom,
            wlen,
        )

        return fp

    @pytest.fixture
    def whh_rel_height(
        self,
    ) -> float:
        return 0.5

    @pytest.fixture
    def pb_rel_height(
        self,
    ) -> float:
        return 1.0

    @pytest.fixture
    def whh(
        self,
        mpm: MapPeaksMixin,
        amp_bcorr: Series[pd.Float64Dtype],
        fp: DataFrame[FindPeaks],
        whh_rel_height: float,
    ) -> DataFrame[WHH]:
        whh = DataFrame[WHH](
            mpm.width_df_factory(amp_bcorr, fp, whh_rel_height, None, "whh")
        )

        return whh

    @pytest.fixture
    def pb(
        self,
        mpm: MapPeaksMixin,
        amp_bcorr: FloatArray,
        fp: DataFrame[FindPeaks],
        pb_rel_height: float,
    ) -> DataFrame[PeakBases]:
        pb_ = mpm.width_df_factory(amp_bcorr, fp, pb_rel_height, None, "pb")
        pb = DataFrame[PeakBases](pb_)

        return pb

    @pytest.fixture
    def mpp(
        self,
    ):
        mpp = MapPeakPlots()
        return mpp

    @pytest.fixture
    def peak_amp_colname(self) -> Literal["amp"]:
        return "amp"


class TestMapPeaksMixin(TestMapPeaksFix):
    def test_set_fp(
        self,
        fp: DataFrame[FindPeaks],
    ):
        fp = fp.reset_index(drop=True).rename_axis(index="idx")

        import pytest

        pytest.set_trace()

        try:
            FindPeaks.validate(fp, lazy=True)
        except pa.errors.SchemaError as e:
            e.add_note(f"\n{e.data}")
            e.add_note(f"\n{e.failure_cases}")

    def test_set_whh(
        self,
        whh: DataFrame[WHH],
    ) -> None:
        import pytest

        pytest.set_trace()
        WHH(whh)

    def test_set_pb(
        self,
        pb: DataFrame[PeakBases],
    ) -> None:
        PeakBases(pb)

    def test_compare_input_amp(
        self,
        main_peak_widths_amp_input,
        amp_bcorr,
        main_chm_asschrom_fitted_pk
    ):
        
        df = pl.DataFrame({"main":main_peak_widths_amp_input, "mine":amp_bcorr}).with_columns(diff=pl.col('main')-pl.col('mine'))
        
        breakpoint()
        
    
    def test_set_peak_map(
        self,
        pm: DataFrame[PeakMap],
    ) -> None:
        import pytest

        pytest.set_trace()
        PeakMap(pm)

    def test_peak_map_compare(self, pm, main_peak_map, main_chm_asschrom_fitted_pk):

        mpm = pl.from_pandas(main_peak_map)
        pm = pl.from_pandas(pm)


        mpm = mpm.with_columns(
            ps.matches("^*._left$|^*._right$").cast(pl.Int64),
        )

        pm = (
            pm.drop(
                [
                    "prom",
                    "prom_right",
                    "prom_left",
                    "time",
                    "amp",
                    "whh_rel_height",
                    "pb_rel_height",
                    "time_idx",
                ]
            )
            .rename(
                {
                    "whh_width": "whh",
                }
            )
            .with_columns(
                ps.matches("^*._left$|^*._right$").cast(pl.Int64),
                p_idx=pl.col("p_idx").cast(pl.UInt32),
            )
        )


        pm_ = pm.with_columns(source=pl.lit("mine")).melt(id_vars=["source", "p_idx"])
        mpm_ = (
            mpm.with_row_index("p_idx")
            .with_columns(source=pl.lit("main"))
            .melt(id_vars=["source", "p_idx"])
        )
        breakpoint()
        df = pl.concat([pm_, mpm_])
        df = (df.pivot(
            columns="source", values="value", index=["p_idx", "variable"]
        )
            .with_columns(
                diff=(pl.col("mine") - pl.col("main")).abs(),
                tol_perc=pl.lit(0.05),
                av_hz=pl.sum_horizontal('mine','main')/2,
            )
            .with_columns(
                tol_limit=pl.col("av_hz") * pl.col("tol_perc"),
            )
            .with_columns(
                tol_pass=pl.col("diff") <= pl.col("tol_limit"),
            )
        )
        
        
        # polt.assert_frame_equal(mpm, pm.drop(['p_idx','time_idx']))
        breakpoint()
        pass


class TestMapPeaks(TestMapPeaksFix):
    def test_map_peaks(
        self,
        pm: DataFrame[PeakMap],
    ) -> None:
        PeakMap(pm, lazy=True)


class TestDFrameChecks:
    @pytest.fixture
    def dfc(
        self,
    ) -> DFrameChecks:
        dfc = DFrameChecks()

        return dfc

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
        dfc: DFrameChecks,
        not_df: Any,
    ) -> None:
        try:
            dfc._check_df(not_df)
        except TypeError:
            pass

    def test_check_df_empty(
        self,
        dfc: DFrameChecks,
        empty_df: pd.DataFrame,
    ) -> None:
        try:
            dfc._check_df(empty_df)
        except ValueError:
            pass


class TestMapPeakPlots(TestMapPeaksFix, TestDFrameChecks):
    def test_signal_plot(
        self,
        mpp: MapPeakPlots,
        bcorred_signal_df: DataFrame[SignalDFBCorr],
        time_colname: Literal["time"],
        bcorr_colname: Literal["amp_corrected"],
        mp,
    ) -> None:
        mpp._plot_signal_factory(bcorred_signal_df, time_colname, bcorr_colname)
        plt.show()

    def test_plot_peaks(
        self,
        mpp: MapPeakPlots,
        mpm: MapPeaksMixin,
        bcorred_signal_df: DataFrame[SignalDFBCorr],
        pm: DataFrame[PeakMap],
        time_colname: Literal["time"],
        bcorr_colname: Literal["amp_corrected"],
    ) -> None:
        PeakMap(pm)

        mpp._plot_signal_factory(bcorred_signal_df, time_colname, bcorr_colname)

        mpp._plot_peaks(
            pm,
            mpm._ptime_col,
            mpm._pmaxima_col,
        )

        plt.show()

    def test_plot_whh(
        self,
        mpp: MapPeakPlots,
        mpm: MapPeaksMixin,
        pm: DataFrame[PeakMap],
        bcorred_signal_df: DataFrame[SignalDFBCorr],
        time_colname: str,
        bcorr_colname: str,
        timestep: float,
    ) -> None:
        mpp._plot_signal_factory(bcorred_signal_df, time_colname, bcorr_colname)

        mpp.plot_whh(pm, mpm._whh_h_col, mpm._whh_l_col, mpm._whh_r_col, timestep)

        plt.show()

    def test_pb_plot(
        self,
        bcorred_signal_df: DataFrame[SignalDFBCorr],
        bcorr_colname: str,
        time_colname: str,
        mpp: MapPeakPlots,
        mpm: MapPeaksMixin,
        pm: DataFrame[PeakMap],
        timestep: float,
    ) -> None:
        mpp._plot_signal_factory(
            bcorred_signal_df,
            time_colname,
            bcorr_colname,
        )
        mpp._plot_peaks(
            pm,
            mpm._ptime_col,
            mpm._pmaxima_col,
        )
        mpp.plot_whh(pm, mpm._whh_h_col, mpm._whh_l_col, mpm._whh_r_col, timestep)

        mpp.plot_bases(
            pm,
            mpm._pb_h_col,
            mpm._pb_l_col,
            mpm._pb_r_col,
            timestep,
            plot_kwargs={"marker": "+"},
        )

        plt.show()
