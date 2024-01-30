from numpy import float64, int64
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias

import numpy as np
import pandas as pd
import pandera as pa
from matplotlib.axes import Axes as Axes
from pandera.typing import DataFrame, Series
from scipy import signal  # type: ignore

from hplc_py.hplc_py_typing.typed_dicts import FindPeaksKwargs

from hplc_py.hplc_py_typing.hplc_py_typing import (
    WHH,
    FindPeaks,
    PeakBases,
    PeakMap,
)


from typing import Tuple

from hplc_py.io_validation import IOValid


PPD: TypeAlias = Tuple[NDArray[float64], NDArray[int64], NDArray[int64]]

@dataclass
class MapPeaks(IOValid):
    """
    Use to map peaks, i.e. locate maxima, whh, peak bases for a given set of user inputs. Includes plotting functionality inherited from MapPeakPlots
    """
    _idx_name: Literal["idx"] = "idx"
    _pidx_col: Literal["p_idx"] = "p_idx"
    _ptime_col: Literal["time"] = "time"
    _ptime_idx_col: Literal["time_idx"] = "time_idx"
    _pmaxima_col: Literal["amp"] = "amp"
    _prom_col: Literal["prom"] = "prom"
    _prom_lb_col: Literal["prom_left"] = "prom_left"
    _prom_rb_col: Literal["prom_right"] = "prom_right"
    _whh_rel_height_col: Literal["whh_rel_height"] = "whh_rel_height"
    _whh_h_col: Literal["whh_height"] = "whh_height"
    _whh_w_col: Literal["whh_width"] = "whh_width"
    _whh_l_col: Literal["whh_left"] = "whh_left"
    _whh_r_col: Literal["whh_right"] = "whh_right"
    _pb_rel_height_col: Literal["pb_rel_height"] = "pb_rel_height"
    _pb_h_col: Literal["pb_height"] = "pb_height"
    _pb_w_col: Literal["pb_width"] = "pb_width"
    _pb_l_col: Literal["pb_left"] = "pb_left"
    _pb_r_col: Literal["pb_right"] = "pb_right"

    @pa.check_types
    def _set_findpeaks(
        self,
        amp: Series[float64],
        time: Series[float64],
        timestep: float,
        prominence: float,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ) -> DataFrame[FindPeaks]:
        # 'denormalize' the prominence input to put it on the scale of amp
        
        for n, s in {"amp":amp, "time": time}.items():
            self.check_container_is_type(s, pd.Series, float, n)
        
        for n, f in {'timestep':timestep, "prom":prominence}.items():
            self.check_scalar_is_type(f, float, n)
            
        if wlen:
            self.check_scalar_is_type(wlen, int, 'wlen') 
        
        prom_ = prominence * amp.max()

        p_idx, _dict = signal.find_peaks(
            amp,
            prominence=prom_,
            wlen=wlen,
            **find_peaks_kwargs,
        )

        fp_: pd.DataFrame = (
            pd.DataFrame(
                {
                    self._ptime_idx_col: p_idx,
                    self._ptime_col: time[p_idx],
                    self._pmaxima_col: amp[p_idx],
                    **_dict,
                }
            )
            .reset_index(drop=True)
            .reset_index(names=self._pidx_col)
            .rename_axis(index=self._idx_name)
        )

        fp_ = fp_.rename(
            {
                "prominences": self._prom_col,
                "left_bases": self._prom_lb_col,
                "right_bases": self._prom_rb_col,
            },
            axis=1,
        )

        fp = DataFrame[FindPeaks](fp_)
        
        return fp

    def width_df_factory(
        self,
        amp: Series[float64],
        peak_time_idx: NDArray[int64],
        peak_prom_data: PPD,
        rel_height: float,
        timestep: float,
        wlen: Optional[int] = None,
        prefix: str = "width",
    ) -> pd.DataFrame:
        """
        width is calculated by first identifying a height to measure the width at, calculated as:
        (peak height) - ((peak prominance) * (relative height))

        width half height, width half height height
        measure the width at half the hieght for a better approximation of
        the latent peak

        this measurement defines the 'scale' paramter of the skewnorm distribution
        for the signal peak reconstruction

        :prefix: is used to prefix the column labels, i.e. measured at half is 'whh'
        """

        rel_h_key = prefix + "_rel_height"
        w_key = prefix + "_width"
        h_key = prefix + "_height"
        h_left_idx_key = prefix + "_left" + "_idx"
        h_right_idx_key = prefix + "_right" + "_idx"
        h_left_time_key = prefix + "_left" + "_time" 
        h_right_time_key = prefix + "_right" + "_time"
        

        w, h, left_ips, right_ips = signal.peak_widths(
            amp,
            peak_time_idx,
            rel_height,
            peak_prom_data,
            wlen,
        )

        wdf_: pd.DataFrame = pd.DataFrame().rename_axis(index="idx")

        wdf_[rel_h_key] = [rel_height] * len(peak_time_idx)

        wdf_[w_key] = w
        wdf_[h_key] = h
        wdf_[h_left_idx_key] = left_ips
        wdf_[h_right_idx_key] = right_ips
        wdf_[h_left_time_key] = left_ips * timestep
        wdf_[h_right_time_key] = right_ips * timestep

        
        wdf: pd.DataFrame = wdf_.reset_index(names=self._pidx_col).rename_axis(
            index=self._idx_name
        )
        
        return wdf

    @pa.check_types
    def _set_peak_map(
        self,
        fp: DataFrame[FindPeaks],
        whh: DataFrame[WHH],
        pb: DataFrame[PeakBases],
    ) -> DataFrame[PeakMap]:
        pm_ = pd.concat(
            [
                fp,
                whh.drop([self._pidx_col], axis=1),
                pb.drop([self._pidx_col], axis=1),
            ],
            axis=1,
        )
        try:
            pm: DataFrame[PeakMap] = DataFrame[PeakMap](pm_)

            PeakMap.validate(pm_, lazy=True)

        except pa.errors.SchemaError as e:
            raise e
        else:
            pm: DataFrame[PeakMap] = DataFrame[PeakMap](pm_)

        return pm
    
    def get_peak_prom_data(
            self,
            fp: DataFrame[FindPeaks],   
        )->PPD:
        peak_prom_data: PPD = tuple(
            [
                fp[self._prom_col].to_numpy(float64),
                fp[self._prom_lb_col].to_numpy(np.int64),
                fp[self._prom_rb_col].to_numpy(np.int64),
            ]
            ) #type: ignore
        return peak_prom_data


    @pa.check_types()
    def map_peaks(
        self,
        amp: Series[float64],
        time: Series[float64],
        timestep: float,
        prominence: float = 0.01,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ) -> DataFrame[PeakMap]:
        """
        Map An input signal with peaks, providing peak height, prominence, and width data.
        """

        fp = self._set_findpeaks(
            amp=amp,
            time=time,
            timestep=timestep,
            prominence=prominence,
            wlen=wlen,
            find_peaks_kwargs=find_peaks_kwargs
            )

        peak_prom_data = self.get_peak_prom_data(
            fp,
            
        )
        
        
        peak_time_idx = fp[self._ptime_idx_col].to_numpy(np.int64)
        
        whh = self.width_df_factory(amp=amp,
                                    peak_time_idx=peak_time_idx,
                                    peak_prom_data=peak_prom_data,
                                    rel_height=0.5,
                                    timestep=timestep,
                                    wlen=None,
                                    prefix="whh")
        
        whh = DataFrame[WHH](whh)

        pb = self.width_df_factory(
                                amp=amp,
                                peak_time_idx=peak_time_idx,
                                peak_prom_data=peak_prom_data,
                                rel_height=1,
                                timestep=timestep,
                                wlen=None,
                                prefix='pb'
                            )
        
        pb = DataFrame[PeakBases](pb)
        
        peak_map = self._set_peak_map(
            fp,
            whh,
            pb,
        )
        
        return DataFrame[PeakMap](peak_map)
