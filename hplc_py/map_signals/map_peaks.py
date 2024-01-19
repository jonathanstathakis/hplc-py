import pandera.extensions as extensions
from pandera.api.pandas.model_config import BaseConfig
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal, Optional, TypedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
from matplotlib.axes import Axes as Axes
from matplotlib.colors import Colormap, ListedColormap
from numpy.typing import ArrayLike
from pandera.typing import DataFrame, Series, Index
from scipy import signal  # type: ignore

from hplc_py.hplc_py_typing.hplc_py_typing import (
    FloatArray,
    rgba,
    BaseDF,
)
from hplc_py.map_signals.checks import DFrameChecks


class FindPeaksKwargs(TypedDict, total=False):
    height: Optional[float | ArrayLike]
    threshold: Optional[float | ArrayLike]
    distance: Optional[float]
    width: Optional[float | ArrayLike]
    plateau_size: Optional[float | ArrayLike]

class FindPeaks(BaseDF):
    p_idx: pd.Int64Dtype
    time_idx: pd.Int64Dtype
    time: pd.Float64Dtype
    amp: pd.Float64Dtype
    prom: pd.Float64Dtype
    prom_left: pd.Int64Dtype
    prom_right: pd.Int64Dtype    

class WHH(BaseDF):
    p_idx: pd.Int64Dtype
    whh_rel_height: pd.Float64Dtype
    whh_width: pd.Float64Dtype
    whh_height: pd.Float64Dtype
    whh_left: pd.Float64Dtype
    whh_right: pd.Float64Dtype

class PeakBases(BaseDF):
    p_idx: pd.Int64Dtype
    pb_rel_height: pd.Float64Dtype
    pb_width: pd.Float64Dtype
    pb_height: pd.Float64Dtype
    pb_left: pd.Float64Dtype
    pb_right: pd.Float64Dtype

@extensions.register_check_method(statistics=['col_a','col_b'])
def col_a_less_than_col_b(df, *, col_a: str, col_b: str):
    return df[col_a]<df[col_b]

class PeakMap(
    PeakBases,
    WHH,
    FindPeaks,
              ):
    
    class Config(BaseDF.Config):
        col_a_less_than_col_b = {"col_a":"whh_width","col_b":"pb_width"}
        


class MapPeakPlots(DFrameChecks):
    
    def __init__(
        self
    ):
        self._cmap: ListedColormap = mpl.colormaps["Set1"]

    def _check_keys_in_index(
        self,
        keys: list[str],
        index: pd.Index,
    ) -> None:
        keys_: Series[str] = Series(keys)

        if not (key_mask := keys_.isin(index)).any():
            raise ValueError(
                f"The following provided keys are not in index: {keys_[~key_mask].tolist()}\npossible keys: {index}"
            )

    def _plot_signal_factory(
        self,
        df: DataFrame,
        x_colname: str,
        y_colname: str,
        ax: Optional[Axes] = None,
        line_kwargs: dict = {},
    ) -> Axes:
        if not ax:
            ax = plt.gca()

        self._check_df(df)
        self._check_keys_in_index([x_colname, y_colname], df.columns)

        sig_x = df[x_colname]
        sig_y = df[y_colname]

        ax.plot(sig_x, sig_y, label="bcorred_signal", **line_kwargs)
        ax.legend()

        return ax

    def _resample_cmap(
        self,
        cmap_l: int,
    ) -> None:
        resampled = self._cmap.resampled(cmap_l)
        
        self._cmap: ListedColormap = resampled

    def _plot_peaks(
        self,
        df: DataFrame,
        x_colname: str,
        y_colname: str,
        label: Optional[str] = "max",
        ax: Optional[Axes] = None,
        plot_kwargs: dict[str, Any] = {},
    ) -> Axes:
        
        """
        Plot peaks from the peak map, x will refer to the time axis, y to amp.
        """
        self._check_df(df)
        self._check_keys_in_index([x_colname, y_colname], df.columns)

        if not ax:
            ax = plt.gca()

        if len(df)>1 and len(self._cmap.colors)==1:
            self._resample_cmap(len(df))

        for i, s in df.iterrows():
            label_ = f"{label}_{i}"

            color = self._cmap.colors[i]  # type: ignore

            ax = self._plot_peak_factory(
                s[x_colname],
                s[y_colname],
                label_,
                color,
                ax,
                plot_kwargs,
            )
        
        return ax

    def _plot_peak_factory(
        self,
        x: FloatArray,
        y: FloatArray,
        label: str,
        color: rgba,
        ax: Optional[Axes] = None,
        plot_kwargs: dict[str, Any] = {},
    ) -> Axes:
        
        if not ax:
            ax = plt.gca()

        ax.plot(
            x,
            y,
            label=label,
            c=color,
            marker="o",
            linestyle="",
            **plot_kwargs,
        )
        ax.legend()
        ax.plot()

        return ax

    def _plot_widths(
        self,
        df: DataFrame,
        timestep: float,
        wh_key: str,
        left_ips_key: str,
        right_ips_key: str,
        kind: Optional[Literal["line", "marker"]] = "marker",
        ax: Optional[Axes] = None,
        label: Optional[str] = "width",
        plot_kwargs: dict = {},
    ):
        """
        Main interface for plotting the widths. Wrap this in an outer function for specific measurements i.e. whh, bases
        """

        self._check_df(df)
        self._check_keys_in_index([left_ips_key, right_ips_key], df.columns)

        if not self._cmap:
            self._resample_cmap(len(df))

        ls = ""

        if kind == "line":
            ls = "-"

        if not ax:
            ax = plt.gca()

        s: pd.Series
        i: Hashable
        for i, s in df.iterrows():
            
            s_ = Series[float](s)
            color = self._cmap.colors[int(i)]

            ax = self._plot_width_factory(
                s_,
                i,
                ls,
                wh_key,
                left_ips_key,
                right_ips_key,
                color,
                timestep,
                ax,
                plot_kwargs,
                label,
            )
        return ax

    def _plot_width_factory(
        self,
        s: Series[float],
        idx: Hashable,
        ls: Literal["", "-"],
        y_key: str,
        left_key: str,
        right_key: str,
        color: rgba,
        timestep: float = 1,
        ax: Optional[Axes] = None,
        plot_kwargs: dict = {},
        label: Optional[str] = None,
    ):
        """
        For plotting an individual peaks widths measurements(?)

        Use timestep if signal x axis has a different unit to the calculated ips i.e. using units rather than idx
        """

        if isinstance(s, pd.Series):
            if s.empty:
                raise ValueError("series is empty")
        else:
            raise TypeError(f"df expected to be Dataframe, got {type(s)}")

        self._check_keys_in_index(
            [y_key, left_key, right_key],
            s.index,
        )

        if not ax:
            ax = plt.gca()

        color = self._cmap.colors[idx]
        label = f"{label}_{idx}"

        s[[left_key, right_key]] = s[[left_key, right_key]].mul(timestep)

        self._draw_widths(
            s,
            left_key,
            right_key,
            y_key,
            color,
            label,
            ls,
            ax,
            plot_kwargs,
        )

        return ax

    def _draw_widths(
        self,
        s: pd.Series,
        left_x_key: str,
        right_x_key: str,
        y_key: str,
        color: rgba,
        label: str,
        ls: Literal["", "-"],
        ax: Axes,
        plot_kwargs: dict = {},
    ):
        marker = "x"

        pkwargs = plot_kwargs.copy()

        if "marker" in pkwargs:
            marker = pkwargs.pop("marker")

        if ls == "-":
            marker = ""

        ax.plot(
            [s[left_x_key], s[right_x_key]],
            [s[y_key], s[y_key]],
            c=color,
            label=label,
            marker=marker,
            ls=ls,
            **pkwargs,
        )
        ax.legend()

        return ax

    def plot_whh(
        self,
        pm_df: DataFrame[PeakMap],
        y_colname: str,
        left_colname: str,
        right_colname: str,
        timestep: float,
        ax: Optional[Axes] = None,
        kind: Optional[Literal["line", "marker"]] = "marker",
        plot_kwargs: dict = {},
    ) -> Axes:
        """
        Create whh plot specifically.
        """

        label = "whh"

        ax = self._plot_widths(
            pm_df,
            timestep,
            y_colname,
            left_colname,
            right_colname,
            kind,
            ax,
            label,
            plot_kwargs,
        )

        return ax

    def plot_bases(
        self,
        pm_df: DataFrame[PeakMap],
        y_colname: str,
        left_colname: str,
        right_colname: str,
        timestep: float,
        ax: Optional[Axes] = None,
        kind: Optional[Literal["line", "marker"]] = "marker",
        plot_kwargs: dict = {},
    ) -> Axes:
        """
        Create whh plot specifically.
        """

        label = "bases"

        ax = self._plot_widths(
            pm_df,
            timestep,
            y_colname,
            left_colname,
            right_colname,
            kind,
            ax,
            label,
            plot_kwargs,
        )
        return ax


@dataclass
class MapPeaksMixin:
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

    """
    This interface only needs an amp and time series, rest is self contained. Dont worry about intermediate steps, just pump out everything, write a seperate class to interface for plotting. Just dump everything into one table, subset elsewhere
    """

    @pa.check_types()
    def _set_fp_df(
        self,
        amp: Series[pd.Float64Dtype],
        time: Series[pd.Float64Dtype],
        prominence: float,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ) -> DataFrame[FindPeaks]:
        
        # 'denormalize' the prominence input to put it on the scale of amp
        prom_ = prominence * amp.max()

        peak_idx, _dict = signal.find_peaks(
            amp,
            prominence=prom_,
            wlen=wlen,
            **find_peaks_kwargs,
        )

        fp_: pd.DataFrame = pd.DataFrame(
            {
                self._ptime_idx_col: peak_idx,
                self._ptime_col: time[peak_idx],
                self._pmaxima_col: amp[peak_idx],
                **_dict,
            }
        ).reset_index(drop=True).reset_index(names=self._pidx_col).rename_axis(index=self._idx_name)
        
        fp_ = fp_.rename(
            {
                "prominences": self._prom_col,
                "left_bases": self._prom_lb_col,
                "right_bases": self._prom_rb_col,
            },
            axis=1,
        )
        
        fp_ = fp_.astype(
            {
                self._pidx_col: pd.Int64Dtype(),
                self._ptime_col: pd.Float64Dtype(),
                self._ptime_idx_col: pd.Int64Dtype(),
                self._pmaxima_col: pd.Float64Dtype(),
                self._prom_col: pd.Float64Dtype(),
                self._prom_lb_col: pd.Int64Dtype(),
                self._prom_rb_col: pd.Int64Dtype(),
            }
        )
        
        try:
            fp: DataFrame[FindPeaks] = DataFrame[FindPeaks](fp_)
        except pa.errors.SchemaError as e:
            err_data = e.data
            err_fc = e.failure_cases
            
            e.add_note(err_data.to_markdown())
            e.add_note(str(err_fc))
            
            raise e
        return fp

    def width_df_factory(
        self,
        amp: Series[pd.Float64Dtype],
        fp_df: DataFrame[FindPeaks],
        rel_height: float,
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
        h_left_key = prefix + "_left"
        h_right_key = prefix + "_right"

        peak_prom_data = tuple(
            [
                fp_df[self._prom_col].to_numpy(np.float64),
                fp_df[self._prom_lb_col].to_numpy(np.int64),
                fp_df[self._prom_rb_col].to_numpy(np.int64),
            ]
        )

        time_idx = fp_df[self._ptime_idx_col].to_numpy(np.int64)

        w, h, left_ips, right_ips = signal.peak_widths(
            amp,
            time_idx,
            rel_height,
            peak_prom_data,
            wlen,
        )

        wdf_: pd.DataFrame = pd.DataFrame().rename_axis(index='idx')

        wdf_[rel_h_key] = [rel_height] * len(time_idx)

        wdf_[w_key] = w
        wdf_[h_key] = h
        wdf_[h_left_key] = left_ips
        wdf_[h_right_key] = right_ips

        wdf_: pd.DataFrame = wdf_.astype(
            {
                rel_h_key: pd.Float64Dtype(),
                w_key: pd.Float64Dtype(),
                h_key: pd.Float64Dtype(),
                h_left_key: pd.Float64Dtype(),
                h_right_key: pd.Float64Dtype(),
            }
        )
        
        wdf_: pd.DataFrame = wdf_.reset_index(names=self._pidx_col).rename_axis(index=self._idx_name)
        
        
        
        wdf: pd.DataFrame = wdf_

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
            import pytest; pytest.set_trace()
            
            # e.add_note(str(e.data)
            raise e
        else:
            pm: DataFrame[PeakMap] = DataFrame[PeakMap](pm_)
        
        return pm


@dataclass
class MapPeaks(MapPeaksMixin, MapPeakPlots):
    
    """
    Use to map peaks, i.e. locate maxima, whh, peak bases for a given set of user inputs. Includes plotting functionality inherited from MapPeakPlots
    """
    def __post_init__(
        self
    ):
        MapPeakPlots.__init__(self)

    @pa.check_types()
    def map_peaks(
        self,
        amp: Series[pd.Float64Dtype],
        time: Series[pd.Float64Dtype],
        prominence: float = 0.01,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ) -> DataFrame[PeakMap]:
        """
        Map An input signal with peaks, providing peak height, prominence, and width data.
        """

        fp = self._set_fp_df(amp, time, prominence, wlen, find_peaks_kwargs)

        whh = self.width_df_factory(amp, fp, 0.5, prefix="whh")
        whh = DataFrame[WHH](whh)

        pb = self.width_df_factory(amp, fp, 1, prefix="pb")
        pb = DataFrame[PeakBases](pb)

        peak_map = self._set_peak_map(
            fp,
            whh,
            pb,
        )

        return DataFrame[PeakMap](peak_map)
