import numpy as np
import pandas as pd
import pytest
from pandera.typing.pandas import DataFrame, Series
import matplotlib.pyplot as plt
from hplc_py.baseline_correct.correct_baseline import SignalDFBCorr
from hplc_py.hplc_py_typing.hplc_py_typing import (
    FloatArray,
    SignalDF,
)
from hplc_py.map_signals.map_peaks import MapPeaks, PeakMap
from hplc_py.map_signals.map_windows import (
    MapWindowPlots,
    MapWindows,
    MapWindowsMixin,
    PeakWindows,
    PWdwdTime,
    WindowedSignalDF,
    WindowedTime,
)

OutPeakDFAssChrom = PeakMap


pd.options.display.precision = 9


pd.options.display.max_columns = 50

class TestMapWindowsFix:
    
    @pytest.fixture
    def mwm(
        self,
    )-> MapWindowsMixin:
        
        mwm = MapWindowsMixin()
        return mwm
    
    @pytest.fixture
    def mw(
        self,
    )->MapWindows:
        mw = MapWindows()
        
        return mw
    @pytest.fixture
    def pm(
        self,
        amp_bcorr: Series[pd.Float64Dtype],
        time: Series[pd.Float64Dtype],
    )-> DataFrame[PeakMap]:
        mp = MapPeaks()
        pm = mp.map_peaks(amp_bcorr, time,)
        return pm
    
    @pytest.fixture
    def time_idx(
        self,
        time: Series[pd.Float64Dtype],
    ):
        time_idx = np.arange(0, len(time), 1)
        
        return time_idx

    @pytest.fixture
    def left_bases(
        self,
        pm: DataFrame[PeakMap],
    )-> Series[pd.Int64Dtype]:
        left_bases: Series[pd.Int64Dtype] = Series[pd.Int64Dtype](pm[PeakMap.pb_left], dtype=pd.Int64Dtype())
        return left_bases
    
    @pytest.fixture
    def right_bases(
        self,
        pm: DataFrame[PeakMap],
    )-> Series[pd.Int64Dtype]:
        right_bases: Series[pd.Int64Dtype] = Series[pd.Int64Dtype](pm[PeakMap.pb_right], dtype=pd.Int64Dtype())
        return right_bases
    
    @pytest.fixture
    def intvls(
        self,
        mwm: MapWindowsMixin,
        left_bases: Series[pd.Int64Dtype],
        right_bases: Series[pd.Int64Dtype],
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
        time: Series[pd.Float64Dtype],
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
        time: Series[pd.Float64Dtype],
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
            pwdwd_time, mwm.pwdt_sc.window_idx
        )
        
        return wdwd_time
    
    @pytest.fixture
    def wt(
        self,
        mw: MapWindows,
        bcorred_signal_df: DataFrame[SignalDF],
        pm: DataFrame[PeakMap],
    )->DataFrame[WindowedTime]:
        
        
        wt = mw._map_windows(pm['pb_left'], pm['pb_right'], bcorred_signal_df['time'])
        
        return wt
    
    @pytest.fixture
    def ws(
        self,
        mw: MapWindows,
        time: Series[pd.Float64Dtype],
        amp_bcorr: Series[pd.Float64Dtype],
        left_bases: Series[pd.Float64Dtype],
        right_bases: Series[pd.Float64Dtype],
    )->DataFrame[WindowedSignalDF]:
        
        ws = mw.window_signal(
            left_bases,
            right_bases,
            time,
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
        wt: DataFrame[WindowedSignalDF],
    )->None:
        
        WindowedTime(wt, lazy=True)
        
    def test_ws(
        self,
        ws: DataFrame[WindowedSignalDF],
    )->None:
        
        WindowedSignalDF(ws)
        
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
        ws: DataFrame[WindowedSignalDF],
        pm: DataFrame[PeakMap],
    )->None:
        mp._plot_signal_factory(ws, mp._ptime_col, 'amp')
        
        height = ws[mwp.ws_sch.amp].max()
        mwp.plot_windows(
            ws,
            height,
        )
        
        mp._plot_peaks(
            pm,
            str(mp._ptime_col),
            str(mp._pmaxima_col),
        )
        

        plt.show()

        import pytest; pytest.set_trace()