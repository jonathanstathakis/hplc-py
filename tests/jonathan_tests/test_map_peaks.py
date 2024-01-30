from matplotlib.figure import Figure
from numpy import float64, int64
from numpy.typing import NDArray

import polars as pl
import polars.selectors as ps
import polars.testing as polt

from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandera.typing import Series, DataFrame
import pandera as pa

from hplc_py.baseline_correct.correct_baseline import SignalDFBCorr

from hplc_py.io_validation import IOValid

from hplc_py.hplc_py_typing.hplc_py_typing import (
    WHH,
    FindPeaks,
    PeakBases,
    PeakMap,
)

from hplc_py.map_signals.map_peaks.map_peaks import MapPeaks, PPD
from hplc_py.map_signals.map_peaks.map_peaks_viz import PeakMapViz
from hplc_py.show import PlotSignal
from matplotlib.axes import Axes

pl.Config(set_tbl_cols=50)


class TestMapPeaksFix:
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
        amp_bcorr: Series[float64],
        time_pd_series: Series[float64],
        timestep: float,
        prom: float,
        mp: MapPeaks,
        wlen: None,
    ) -> DataFrame[FindPeaks]:
        fp = mp._set_findpeaks(
            amp=amp_bcorr,
            time=time_pd_series,
            timestep=timestep,
            prominence=prom,
            wlen=wlen,
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
    def ptime_idx_col(
        self,
    ):
        return str(FindPeaks.time_idx)

    @pytest.fixture
    def ptime_idx(
        self,
        fp: DataFrame[FindPeaks],
        ptime_idx_col: str,
    ) -> NDArray[int64]:
        return fp[ptime_idx_col].to_numpy(int64)

    @pytest.fixture
    def ppd(self, mp: MapPeaks, fp: DataFrame[FindPeaks]) -> PPD:
        ppd = mp.get_peak_prom_data(fp)
        return ppd

    @pytest.fixture
    def whh(
        self,
        mp: MapPeaks,
        amp_bcorr: Series[float64],
        ptime_idx: NDArray[int64],
        ppd: PPD,
        whh_rel_height: float,
        timestep: float,
    ) -> DataFrame[WHH]:
        whh = DataFrame[WHH](
            mp.width_df_factory(
                amp=amp_bcorr,
                peak_time_idx=ptime_idx,
                peak_prom_data=ppd,
                rel_height=whh_rel_height,
                timestep=timestep,
                wlen=None,
                prefix="whh",
            )
        )

        return whh

    @pytest.fixture
    def pb(
        self,
        mp: MapPeaks,
        amp_bcorr: Series[float64],
        ptime_idx: NDArray[int64],
        pb_rel_height: float,
        ppd: PPD,
        timestep: float,
    ) -> DataFrame[PeakBases]:
        pb_ = mp.width_df_factory(
            amp=amp_bcorr,
            peak_time_idx=ptime_idx,
            peak_prom_data=ppd,
            rel_height=pb_rel_height,
            timestep=timestep,
            wlen=None,
            prefix="pb",
        )

        pb = DataFrame[PeakBases](pb_)

        return pb


class TestMapPeaks(TestMapPeaksFix):
    def test_set_fp(
        self,
        fp: DataFrame[FindPeaks],
    ):
        fp_ = fp.reset_index(drop=True).rename_axis(index="idx")

        try:
            FindPeaks.validate(fp_, lazy=True)
        except pa.errors.SchemaError as e:
            e.add_note(f"\n{e.data}")
            e.add_note(f"\n{e.failure_cases}")

    def test_set_whh(
        self,
        whh: DataFrame[WHH],
    ) -> None:
        WHH(whh)

    def test_set_pb(
        self,
        pb: DataFrame[PeakBases],
    ) -> None:
        PeakBases(pb)

    def test_compare_input_amp(
        self, main_peak_widths_amp_input, amp_bcorr, main_chm_asschrom_fitted_pk
    ):
        df = pl.DataFrame({"main": main_peak_widths_amp_input, "mine": amp_bcorr})

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
        
    def test_map_peaks(
        self,
        my_peak_map: DataFrame[PeakMap],
    ) -> None:
        PeakMap(my_peak_map, lazy=True)

    def test_set_peak_map(
        self,
        my_peak_map: DataFrame[PeakMap],
    ) -> None:
        PeakMap(my_peak_map)

    def test_peak_map_compare(self, my_peak_map, main_pm_, timestep: float):
        """
        Compare my peak map to the main module version of the peak map.
        """
        main_peak_map_ = pl.from_pandas(main_pm_)
        my_peak_map = pl.from_pandas(my_peak_map)

        # cast the 'idx' type columns to int64 and suffix with "idx"

        idx_unit_cols = ps.matches("^*._left$|^*._right$")
        main_peak_map_ = (
            main_peak_map_.with_columns(
                idx_unit_cols.cast(pl.Int64).name.suffix("_idx")
            )
            .with_columns(idx_unit_cols.mul(timestep).name.suffix("_time"))
            .drop(idx_unit_cols)
            .with_columns(time=pl.col("time_idx") * timestep)
        )

        my_peak_map = my_peak_map.drop(
            [
                "prom",
                "prom_right",
                "prom_left",
            ]
        )

        my_peak_map = my_peak_map.melt(id_vars=["p_idx"], value_name='mine',)

        main_peak_map = main_peak_map_.melt(id_vars=["p_idx"], value_name='main')

        # inspect to see why the join might fail

        aj_my_to_main = my_peak_map.join(
            main_peak_map, how="anti", on=["p_idx", "variable"]
        )
        aj_main_to_my = main_peak_map.join(
            my_peak_map, how="anti", on=["p_idx", "variable"]
        )

        if not aj_main_to_my.is_empty() or not aj_my_to_main.is_empty():
            raise ValueError(
                f"not all keys are paired.\n\nIn my table, the following are not present in the right:{aj_my_to_main}\n\nIn main table, the following are not present in my table:{aj_main_to_my}"
            )

        df = my_peak_map.join(main_peak_map, on=["p_idx", "variable"], how="left")

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
            iov._check_df(not_df)
        except TypeError:
            pass

    def test_check_df_empty(
        self,
        iov: IOValid,
        empty_df: pd.DataFrame,
    ) -> None:
        try:
            iov._check_df(empty_df)
        except ValueError:
            pass


class TestPeakMapViz(TestMapPeaksFix):
    """
    Test whether the PeakMapViz class correctly plots the peak map values onto the signal.
    
    Works by passing a pre-generated Axes object between the fixtures, iteratively drawing
    onto the Axes to produce the expected overlay plots. `test_chain_plot` produces
    the same thing but via method chaining from the `loaded_pmv` object.
    
    Note: Uses time rather than time_idx as the x-axis units.

    :param TestMapPeaksFix: Contains fixtures shared between all test classes testing
    the `MapPeaks` submodule.
    :type TestMapPeaksFix: TestMapPeaksFix
    """

    @pytest.fixture
    def test_peak_map_viz_fig(self):
        return plt.figure()

    @pytest.fixture
    def test_peak_map_viz_ax(self, test_peak_map_viz_fig):
        return test_peak_map_viz_fig.subplots()

    @pytest.fixture
    def plot_signal(
        self,
        amp_bcorr: DataFrame[SignalDFBCorr],
        time_col: Literal["time"],
        bcorr_colname: Literal["amp_corrected"],
        test_peak_map_viz_ax: Axes,
    ):
        PlotSignal(
            df=amp_bcorr,
            x_colname=time_col,
            y_colname=bcorr_colname,
            label=bcorr_colname,
            ax=test_peak_map_viz_ax,
        )._plot_signal_factory()
        
        return None
        

    @pytest.fixture
    def loaded_pmv(
        self,
        my_peak_map: DataFrame[PeakMap],
        test_peak_map_viz_ax: Axes,
        time_col: str,
    ) -> PeakMapViz:
        
        pmv = PeakMapViz(df=my_peak_map, x_colname=time_col, ax=test_peak_map_viz_ax)

        return pmv

    @pytest.fixture
    def plot_peaks(
        self,
        loaded_pmv: PeakMapViz,
        amp_col: str,
    ) -> Axes:

        loaded_pmv._plot_peaks(y_colname=amp_col)

        return None

    def test_signal_plot(
        self,
        plot_signal,
    ) -> None:
        plt.show()

    def test_plot_peaks(
        self,
        test_peak_map_viz_fig: Figure,
        plot_signal,
        plot_peaks,
    ):
        test_peak_map_viz_fig.show()
        plt.show()

    @pytest.fixture
    def plot_whh(
        self,
        test_peak_map_viz_ax,
        loaded_pmv: PeakMapViz,
    ):
        loaded_pmv.plot_whh(
            y_colname=str(PeakMap.whh_height),
            left_colname=str(PeakMap.whh_left_time),
            right_colname=str(PeakMap.whh_right_time),
            ax=test_peak_map_viz_ax
        )

    def test_plot_whh(
        self,
        plot_signal,
        plot_peaks,
        plot_whh,
    ) -> None:
        plt.show()


    @pytest.fixture
    def plot_pb(
        self,
        test_peak_map_viz_ax,
        loaded_pmv: PeakMapViz,
    ):
        loaded_pmv.plot_bases(
            y_colname=str(PeakMap.pb_height),
            left_colname=str(PeakMap.pb_left_time),
            right_colname=str(PeakMap.pb_right_time),
            ax=test_peak_map_viz_ax
        )
        
    def test_plot_pb(
        self,
        plot_signal,
        plot_peaks,
        plot_whh,
        plot_pb,
    )->None:
        plt.show()
        
    def test_chain_plot(
        self,
        loaded_pmv: PeakMapViz,
        plot_signal,
    )->None:
        """
        Test whether chaining the plot methods produces the expected drawing

        :param loaded_pmv: PeakMapViz object loaded with data
        :type loaded_pmv: PeakMapViz
        :param plot_signal: A fixture plotting the signal on the axis loaded into `loaded_pmv`
        :type plot_signal: None
        """
        (loaded_pmv
         .plot_whh(PeakMap.whh_height, PeakMap.whh_left_time, PeakMap.whh_right_time)
         .plot_bases(PeakMap.pb_height, PeakMap.pb_left_time, PeakMap.pb_right_time)
         )
        
        plt.show()