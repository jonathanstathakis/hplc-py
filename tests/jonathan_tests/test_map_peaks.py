
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandera.typing import DataFrame
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
    )->MapPeaks:
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
        amp_bcorr: FloatArray,
        fp: DataFrame[FindPeaks],
        whh_rel_height: float,
    ) -> DataFrame[WHH]:
        whh = DataFrame[WHH](mpm.width_df_factory(
            amp_bcorr,
            fp,
            whh_rel_height,
            None,
            'whh'
        ))

        return whh
    
    @pytest.fixture
    def pb(
        self,
        mpm: MapPeaksMixin,
        amp_bcorr: FloatArray,
        fp: DataFrame[FindPeaks],
        pb_rel_height: float,
    ) -> DataFrame[PeakBases]:
        
        pb_ = mpm.width_df_factory(
            amp_bcorr,
            fp,
            pb_rel_height,
            None,
            'pb'
        )
        pb = DataFrame[PeakBases](pb_)

        return pb


    @pytest.fixture
    def pm(
        self,
        mpm: MapPeaksMixin,
        fp: DataFrame[FindPeaks],
        whh: DataFrame[WHH],
        pb: DataFrame[PeakBases],
    ) -> DataFrame[PeakMap]:
        
        pm = mpm._set_peak_map(
            fp,
            whh,
            pb,
        )
        return pm

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
        fp = fp.reset_index(drop=True).rename_axis(index='idx')
        
        try:
            FindPeaks.validate(fp, lazy=True)
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
        
    def test_set_peak_map(
        self,
        pm: DataFrame[PeakMap],
    ) -> None:
        PeakMap(pm)
        
class TestMapPeaks(TestMapPeaksFix):
    def test_map_peaks(
        self,
        pm: DataFrame[PeakMap],
    )->None:
        
        PeakMap(pm, lazy=True)


class TestDFrameChecks:
    @pytest.fixture
    def dfc(
        self,
    )->DFrameChecks:
        
        dfc = DFrameChecks()
        
        return dfc
    
    @pytest.fixture
    def empty_df(
        self
    ):
        df = pd.DataFrame()
        return df    
    
    @pytest.fixture
    def not_df(
        self
    ):
       x = 0
       
       return x
    
    def test_check_df_not_df(
        self,
        dfc: DFrameChecks,
        not_df: Any,
    )->None:
        
        try:
            dfc._check_df(not_df)
        except TypeError:
            pass
        
    def test_check_df_empty(
        self,
        dfc: DFrameChecks,
        empty_df: pd.DataFrame,
    )->None:
        
        try:
            dfc._check_df(
                empty_df
            )
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
            plot_kwargs={"marker":"+"}
        )

        plt.show()
