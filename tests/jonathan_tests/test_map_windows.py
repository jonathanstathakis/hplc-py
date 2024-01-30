from numpy import float64, int64

import polars as pl

import numpy as np
import pandas as pd
import pytest
from pandera.typing.pandas import DataFrame, Series
import matplotlib.pyplot as plt

from hplc_py.map_signals.map_peaks.map_peaks import MapPeaks, PeakMap

from hplc_py.map_signals.map_windows import (
    MapWindowPlots,
    MapWindows,
    MapWindowsMixin,
)

from hplc_py.hplc_py_typing.hplc_py_typing import (
    PeakWindows,
    PWdwdTime,
    WindowedSignal,
    WindowedTime,
    SignalDFBCorr,
    )

OutPeakDFAssChrom = PeakMap

class TestMapWindowsFix:
    
    @pytest.fixture
    def mwm(
        self,
    )-> MapWindowsMixin:
        
        mwm = MapWindowsMixin()
        return mwm

    
    @pytest.fixture
    def time_idx(
        self,
        time: Series[float64],
    ):
        time_idx = np.arange(0, len(time), 1)
        
        return time_idx

    @pytest.fixture
    def left_bases(
        self,
        my_peak_map: DataFrame[PeakMap],
    )-> Series[int64]:
        left_bases: Series[int64] = Series[int64](my_peak_map[PeakMap.pb_left], dtype=int64)
        return left_bases
    
    @pytest.fixture
    def right_bases(
        self,
        my_peak_map: DataFrame[PeakMap],
    )-> Series[int64]:
        right_bases: Series[int64] = Series[int64](my_peak_map[PeakMap.pb_right], dtype=int64)
        return right_bases
    
    @pytest.fixture
    def intvls(
        self,
        mwm: MapWindowsMixin,
        left_bases: Series[int64],
        right_bases: Series[int64],
    )->Series[pd.Interval]:
        intvls: Series[pd.Interval] = mwm._interval_factory(left_bases, right_bases)
        return intvls
    
    @pytest.fixture
    def w_idxs(
        self,
        mwm: MapWindowsMixin,
        intvls: Series[pd.Interval],
    )->dict[int, list[int]]:
        w_idxs: dict[int, list[int]] = mwm._label_windows(intvls)
        return w_idxs
    
    @pytest.fixture
    def w_intvls(
        self,
        mwm: MapWindowsMixin,
        intvls: Series[pd.Interval],
        w_idxs: dict[int, list[int]],
    )->dict[int, pd.Interval]:
        w_intvls: dict[int, pd.Interval] = mwm._combine_intvls(intvls, w_idxs)
        return w_intvls

    @pytest.fixture
    def peak_wdws(
        self,
        mwm: MapWindowsMixin,
        time: Series[float64],
        w_intvls: dict[int, pd.Interval],
    )-> DataFrame[PeakWindows]:
        peak_windows = mwm._set_peak_windows(
        w_intvls,
        time,
        )
        
        return peak_windows
    
    @pytest.fixture
    def pwdwd_time(
        self,
        mwm: MapWindowsMixin,
        time: Series[float64],
        peak_wdws: DataFrame[PeakWindows],
    )->DataFrame[PWdwdTime]:
        
        pwdwd_time: DataFrame[PWdwdTime] = mwm._set_peak_wndwd_time(
            time,
            peak_wdws,
        )
        
        return pwdwd_time
    
    @pytest.fixture
    def wdwd_time(
        self,
        mwm: MapWindowsMixin,
        pwdwd_time: DataFrame[PWdwdTime],
    )->DataFrame[WindowedTime]:
        
        wdwd_time: DataFrame[WindowedTime] = mwm._label_interpeaks(
            pwdwd_time, mwm.pwdt_sc.w_idx
        )
        
        return wdwd_time
    
    @pytest.fixture
    def wt(
        self,
        mw: MapWindows,
        bcorred_signal_df_asschrom: DataFrame[SignalDFBCorr],
        my_peak_map: DataFrame[PeakMap],
    )->DataFrame[WindowedTime]:
        
        SignalDFBCorr.validate(bcorred_signal_df_asschrom, lazy=True)
        
        df = bcorred_signal_df_asschrom
        
        wt = mw._map_windows_to_time(my_peak_map['pb_left'], my_peak_map['pb_right'], df['time'])
        
        breakpoint()
        
        return wt
    
    @pytest.fixture
    def ws(
        self,
        mw: MapWindows,
        time_pd_series: Series[float64],
        amp_bcorr: Series[float64],
        left_bases: Series[float64],
        right_bases: Series[float64],
    )->DataFrame[WindowedSignal]:
        
        ws = mw.window_signal(
            left_bases,
            right_bases,
            time_pd_series,
            amp_bcorr,
            )
        
        return ws

        
class TestMapWindows(TestMapWindowsFix):
    
    def test_interval_factory(
        self,
        intvls: Series[pd.Interval],
    )->None:
        
        if not isinstance(intvls, pd.Series):
            raise TypeError("expected pd.Series")
        elif intvls.empty:
            raise ValueError("intvls is empty")
        elif not intvls.index.dtype == np.int64:
            raise TypeError(f"Expected np.int64 index, got {intvls.index.dtype}")
        elif not intvls.index[0]==0:
            raise ValueError("Expected interval to start at 0")
        
        
    def test_label_windows(
        self,
        w_idxs: dict[int, list[int]],
    )->None:
        if not w_idxs:
            raise ValueError('w_idxs is empty')
        if not any(v for v in w_idxs.values()):
            raise ValueError('a w_idx is empty')
        if not isinstance(w_idxs, dict):
            raise TypeError("expected dict")
        elif not all(isinstance(l, list) for l in w_idxs.values()):
            raise TypeError('expected list')
        elif not all(isinstance(x, int) for v in w_idxs.values() for x in v):
            raise TypeError("expected values of window lists to be int")
    
    def test_combine_intvls(
        self,
        w_intvls: dict[str, pd.Interval],
        )->None:
        if not w_intvls:
            raise ValueError("w_intvls is empty")
        if not isinstance(w_intvls, dict):
            raise TypeError("expected dict")
        if not all(isinstance(intvl, pd.Interval) for intvl in w_intvls.values()):
            raise TypeError("expected pd.Interval")
        
    def test_set_peak_windows(
        self,
        peak_wdws: DataFrame[PeakWindows],
    ):
        PeakWindows(peak_wdws)
    
    def test_label_interpeaks(
        self,
        wdwd_time: DataFrame[WindowedTime],
    ):
        WindowedTime.validate(wdwd_time, lazy=True)
        
    def test_map_windows(
        self,
        wt: DataFrame[WindowedSignal],
    )->None:
        
        WindowedTime.validate(wt, lazy=True)
        breakpoint()
        
    def test_ws(
        self,
        ws: DataFrame[WindowedSignal],
    )->None:
        
        WindowedSignal(ws)

        
    def test_compare_window_dfs(
        self,
        ws,
        main_window_df,
    ):
        """
        Compare windowing of the signal between the main and my implementation.
        """
        
        from hplc_py.map_signals.map_windows import MapWindowPlots
        
        mwp = MapWindowPlots()
        
        mwp.plot_windows(ws, ws['amp'].max())
        
        main_window_df = main_window_df.drop(["signal", "estimated_background"]).rename(
            {"window_type": "w_type", "window_id": "w_idx", "signal_corrected": "amp"}
        )
        ws = pl.from_pandas(ws)

        wdws = pl.concat(
            [
                ws.select(["w_type", "w_idx", "time_idx"]).with_columns(
                    source=pl.lit("mine")
                ),
                main_window_df.select(["w_type", "w_idx", "time_idx"]).with_columns(
                    source=pl.lit("main")
                ),
            ]
        )

        wdw_compare = (
            wdws.group_by(["source", "w_type", "w_idx"])
            .agg(start=pl.col("time_idx").first(), end=pl.col("time_idx").last())
            .sort(["start", "source"])
        )

        wdw_compare_diff = wdw_compare.pivot(
            columns="source", values=["start", "end"], index=["w_type", "w_idx"]
        ).with_columns(
            start_diff=pl.col("start_source_main")
            .sub(pl.col("start_source_mine"))
            .abs(),
            end_diff=pl.col("end_source_main").sub(pl.col("end_source_mine")).abs(),
        )

        mwp = MapWindowPlots()
        
        ws = ws.to_pandas()
        ws = ws.astype({
            **{col: pd.StringDtype() for col in ws.select_dtypes(np.object_).columns},
            **{col: float64 for col in ws.select_dtypes(float64).columns},
            **{col: int64 for col in ws.select_dtypes(np.int64).columns},
        }).rename_axis('idx')
        
        mwp.plot_windows(ws, ws['amp'].max())
        plt.show()
        
        
        # polt.assert_frame_equal(main_window_df.drop("estimated_background"), ws)

        
class TestMapWindowPlots(TestMapWindowsFix):
    @pytest.fixture
    def mwp(
        self,
    )->MapWindowPlots:
        mwp = MapWindowPlots()
        return mwp
    
    def test_map_windows_plot(
        self,
        mp: MapPeaks,
        mwp: MapWindowPlots,
        ws: DataFrame[WindowedSignal],
        my_peak_map: DataFrame[PeakMap],
    )->None:
        mp._plot_signal_factory(ws, mp._ptime_col, 'amp')
        
        height = ws[mwp.ws_sch.amp].max()
        mwp.plot_windows(
            ws,
            height,
        )
        
        mp._plot_peaks(
            my_peak_map,
            str(mp._ptime_col),
            str(mp._pmaxima_col),
        )
        

        plt.show()
