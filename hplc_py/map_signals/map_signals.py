"""
1. Identify peaks in chromatographic data
2. clip the chromatogram into discrete peak windows

- Use `scipy.signal.find_peaks` with prominence as main filter

operation # 1 - find peaks
1. normalize before applying algorithm to generalise prominence filter settings
2. obtain the peak maxima indices

operation # 2 - clip the chromatogram into windows

- a window is defined as a region where peaks are overlapping or nearly overlapping.
- a window is identified by measuring the peak width at lowest contour line, peaks with 
overlapping contour lines are defined as influencing each other, therefore in the same window.
- buffer parameter used to control where windows start and finish, their magnitude (?)

TODO:
- [ ] build TypedDict for:
    - [ ] line_kwargs
    - [ ] scatter_kwargs
    - [ ] hline_kwargs
"""


from scipy import signal  # type: ignore
import numpy as np
import warnings
import pandas as pd
import pandera as pa
import pandera.typing as pt
import pandera.typing as pt
import numpy.typing as npt
from pandera.typing import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Colormap
from matplotlib.patches import Rectangle
from scipy.ndimage import label

from dataclasses import dataclass, field
from typing import cast, TypedDict, Optional, Any, Literal

from hplc_py.hplc_py_typing.hplc_py_typing import (
    OutSignalDF_Base,
    OutWindowDF_Base,
    IntArray,
    FloatArray,
    pdInt,
    pdFloat,
    rgba,
)

from hplc_py.misc.misc import LoadData, DataMixin


class FindPeaksKwargs(TypedDict, total=False):
    height: Optional[float | npt.ArrayLike]
    threshold: Optional[float | npt.ArrayLike]
    distance: Optional[float]
    width: Optional[float | npt.ArrayLike]
    plateau_size: Optional[float | npt.ArrayLike]


class FindPeaks(pa.DataFrameModel):
    time_idx: pd.Int64Dtype
    prom: pd.Float64Dtype
    prom_left: pd.Int64Dtype
    prom_right: pd.Int64Dtype


class PeakBases(pa.DataFrameModel):
    pb_rel_height: pd.Float64Dtype
    pb_width: pd.Float64Dtype
    pb_height: pd.Float64Dtype
    pb_left: pd.Float64Dtype
    pb_right: pd.Float64Dtype


class WHH(pa.DataFrameModel):
    whh_rel_height: pd.Float64Dtype
    whh_width: pd.Float64Dtype
    whh_height: pd.Float64Dtype
    whh_left: pd.Float64Dtype
    whh_right: pd.Float64Dtype


class PeakMap(FindPeaks, PeakBases, WHH):
    pass


@dataclass
class MapPeakPlots:
    cmap: None = None

    def __post_init__(self):
        self._cmap: Colormap = mpl.colormaps["Set1"]

    def check_df(
        self,
        df: pd.DataFrame,
    ) -> None:
        if isinstance(df, pd.DataFrame):
            if df.empty:
                raise ValueError("df is empty")
        else:
            raise TypeError(f"df expected to be Dataframe, got {type(df)}\n{df}")

    def check_keys_in_index(
        self,
        keys: list[str],
        index: pd.Index,
    ) -> None:
        keys: Series[str] = pd.Series(keys)

        if not (key_mask := keys.isin(index)).any():
            raise ValueError(
                f"The following provided keys are not in index: {keys[~key_mask].tolist()}\npossible keys: {index}"
            )

    def _plot_signal_factory(
        self,
        df: DataFrame,
        x_colname: str,
        y_colname: str,
        ax: Optional[plt.Axes] = None,
        line_kwargs: Optional[dict] = {},
    ) -> plt.Axes:
        if not ax:
            ax = plt.gca()

        self.check_df(df)
        self.check_keys_in_index([x_colname, y_colname], df.columns)

        sig_x = df[x_colname]
        sig_y = df[y_colname]

        ax.plot(sig_x, sig_y, label="bcorred_signal", **line_kwargs)
        ax.legend()

        return ax

    def _load_cmap(
        self,
        df: DataFrame,
    ) -> None:
        self.cmap: Colormap = self._cmap.resampled(len(df))

    def _plot_peaks(
        self,
        df: DataFrame,
        x_colname: str,
        y_colname: str,
        label: Optional[str] = "max",
        ax: Optional[plt.Axes] = None,
        plot_kwargs: Optional[dict[str, Any]] = {},
    ) -> plt.Axes:
        self.check_df(df)
        self.check_keys_in_index([x_colname, y_colname], df.columns)

        if not ax:
            ax = plt.gca()

        if not self.cmap:
            self._load_cmap(df)

        for i, s in df.iterrows():
            l = f"{label}_{i}"

            color = self._cmap.colors[i]

            ax = self._plot_peak_factory(
                s[x_colname],
                s[y_colname],
                l,
                color,
                ax,
                plot_kwargs,
            )

    def _plot_peak_factory(
        self,
        x: FloatArray,
        y: FloatArray,
        label: str,
        color: rgba,
        ax: Optional[plt.Axes] = None,
        plot_kwargs: Optional[dict[str, Any]] = {},
    ) -> plt.Axes:
        marker = "o"

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
        df: pd.DataFrame,
        timestep: float,
        wh_key: str,
        left_ips_key: str,
        right_ips_key: str,
        kind: Optional[Literal["line", "marker"]] = "marker",
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = "width",
        plot_kwargs: dict = {},
    ):
        """
        Main interface for plotting the widths. Wrap this in an outer function for specific measurements i.e. whh, bases
        """

        self.check_df(df)
        self.check_keys_in_index([left_ips_key, right_ips_key], df.columns)

        if not self.cmap:
            self._load_cmap(df)

        ls = ""

        if kind == "line":
            ls = "-"

        if not ax:
            ax = plt.gca()

        for i, s in df.iterrows():
            color = self._cmap.colors[i]

            ax = self._plot_width_factory(
                s,
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
        s: Series,
        idx: int,
        ls: Literal["", "-"],
        y_key: str,
        left_key: str,
        right_key: str,
        color: rgba,
        timestep: Optional[float] = 1,
        ax: Optional[plt.Axes] = None,
        plot_kwargs: Optional[dict] = {},
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

        self.check_keys_in_index(
            [y_key, left_key, right_key],
            s.index,
        )

        if not ax:
            ax = plt.gca()

        color = self._cmap.colors[idx]
        label = f"{label}_{idx}"

        s[[left_key, right_key]] = s[[left_key, right_key]] * timestep

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
        ax: plt.Axes,
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
        ax: Optional[plt.Axes] = None,
        kind: Optional[Literal["line", "marker"]] = "marker",
        plot_kwargs: dict = {},
    ) -> plt.Axes:
        """
        Create whh plot specifically.
        """

        label = "whh"

        self._plot_widths(
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

    def plot_bases(
        self,
        pm_df: DataFrame[PeakMap],
        y_colname: str,
        left_colname: str,
        right_colname: str,
        timestep: float,
        ax: Optional[plt.Axes] = None,
        kind: Optional[Literal["line", "marker"]] = "marker",
        plot_kwargs: dict = {},
    ) -> plt.Axes:
        """
        Create whh plot specifically.
        """

        label = "bases"

        self._plot_widths(
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


@dataclass
class MapPeaksMixin:
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
        amp: FloatArray,
        time: FloatArray,
        prominence: float,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ) -> DataFrame[FindPeaks]:
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
        )

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
                self._ptime_col: pd.Float64Dtype(),
                self._ptime_idx_col: pd.Int64Dtype(),
                self._pmaxima_col: pd.Float64Dtype(),
                self._prom_col: pd.Float64Dtype(),
                self._prom_lb_col: pd.Int64Dtype(),
                self._prom_rb_col: pd.Int64Dtype(),
            }
        )

        fp: DataFrame[FindPeaks] = DataFrame[FindPeaks](fp_)
        return fp

    
    def width_df_factory(
        self,
        amp: FloatArray,
        fp_df: DataFrame[FindPeaks],
        rel_height: Optional[float] = 1,
        wlen: Optional[int] = None,
        prefix: str = "width_",
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

        w, h, l, r = signal.peak_widths(
            amp,
            time_idx,
            rel_height,
            peak_prom_data,
            wlen,
        )

        df = pd.DataFrame()

        df[rel_h_key] = [rel_height] * len(time_idx)

        df[w_key] = w
        df[h_key] = h
        df[h_left_key] = l
        df[h_right_key] = r

        df = df.astype(
            {
                rel_h_key: pd.Float64Dtype(),
                w_key: pd.Float64Dtype(),
                h_key: pd.Float64Dtype(),
                h_left_key: pd.Float64Dtype(),
                h_right_key: pd.Float64Dtype(),
            }
        )

        return df

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
                whh,
                pb,
            ],
            axis=1,
        )

        pm: DataFrame[PeakMap] = DataFrame[PeakMap](pm_)
        return pm


@dataclass
class MapPeaks(MapPeaksMixin):
    """
    Use to map peaks, i.e. locate maxima, whh, peak bases for a given set of user inputs. Includes plotting functionality inherited from MapPeakPlots
    """

    @pa.check_types()
    def map_peaks(
        self,
        amp: FloatArray,
        time: FloatArray,
        prominence: float = 0.01,
        wlen: Optional[int] = None,
        find_peaks_kwargs: FindPeaksKwargs = {},
    ) -> DataFrame[PeakMap]:
        """
        Map An input signal with peaks, providing peak height, prominence, and width data.
        """

        fp = self._set_fp_df(
            amp,
            time,
            prominence,
            wlen,
            find_peaks_kwargs
        )
        
        whh = self.width_df_factory(amp, fp, 0.5, prefix="whh")
        whh = DataFrame[WHH](whh)
        
        pb = self.width_df_factory(amp, fp, 1, prefix="pb")
        pb = DataFrame[PeakBases](pb)

        peak_map = self._set_peak_map(
            fp,
            whh,
            pb,
        )

        return pt.DataFrame[PeakMap](peak_map)


@dataclass
class MapSignal(LoadData):
    def profile_peaks_assign_windows(
        self,
        prominence: float = 0.01,
        window_rel_height: float = 1,
        buffer: int = 0,
        peak_kwargs: dict = dict(),
    ) -> tuple[pt.DataFrame[PeakMap], pt.DataFrame[OutWindowDF_Base],]:
        R"""
        Profile peaks and assign windows based on user input, return a dataframe containing
        time, signal, transformed signals, peak profile information
        """

        # input validation
        if (rel_height < 0) | (rel_height > 1):
            raise ValueError(f" `rel_height` must be [0, 1].")

        peak_df = self.peak_df_factory(prominence, window_rel_height, peak_kwargs)

        window_df = self.window_df_factory(
            peak_df["rl_left"].to_numpy(np.float64),
            peak_df["rl_right"].to_numpy(np.float64),
            buffer,
        )

        """
        normalize the data.
        
        get the window df as one which tracks which peaks are in which window and contains the metrics 'window area' and 'num peaks'
        
        Thus we end up with 4 tables:
        
        signal table: base and transformed signals.
        peak table: information on each peak.
        window metric table: window area, number of peaks and other metrics.
        
        keys:
        window metric table: window id
        peak table: peak maxima time index, foreign key: window id
        signal table: time idx
        
        key table:
         - window id
         - peak id
         - time idx
        """
        return peak_df, window_df

    def norm_inverse(self, x: FloatArray, x_norm: FloatArray):
        """
        Invert the normalization to express the measure heights in base scale
        """

        x_inv = x_norm * (x.max() - x.min()) + x.min()

        return x_inv

    def mask_subset_ranges(self, ranges: list[IntArray]) -> npt.NDArray[np.bool_]:
        """
        Generate a boolean mask of the peak ranges array which defaults to True, but
        False if for range i, range i+1 is a subset of range i.
        """
        # generate an array of True values
        valid = np.full(len(ranges), True)

        """
        if there is more than one range in ranges, set up two identical nested loops.
        The inner loop skips the first iteration then checks if the values in range i+1
        are a subset of range i, if they are, range i+1 is marked as invalid. 
        A subset is defined as one where the entirety of the subset is contained within
        the superset.
        
        
        """
        if len(ranges) > 1:
            for i, r1 in enumerate(ranges):
                for j, r2 in enumerate(ranges):
                    if i != j:
                        if set(r2).issubset(r1):
                            valid[j] = False
        return valid

    def compute_individual_peak_ranges(
        self,
        amp: FloatArray,
        left_base: FloatArray,
        right_base: FloatArray,
        buffer: int = 0,
    ) -> list[IntArray]:
        """
        calculate the range of each peak based on the left and right base extended by
        the buffer size, restricted to positive values and the length of the intensity
        array.

        Return a list of possible peak ranges
        """

        left_base = np.array(left_base, np.float64)
        right_base = np.array(right_base, np.float64)

        if len(left_base) < 1:
            raise ValueError("left base must be longer than 1")
        if len(right_base) < 1:
            raise ValueError("right base must be longer than 1")

        if left_base.ndim > 1:
            raise ValueError("left_base must be a 1d array")
        if right_base.ndim > 1:
            raise ValueError("right_base must be a 1d array")
        ranges = []

        for l, r in zip(left_base, right_base):
            peak_range = np.arange(int(l - buffer), int(r + buffer), 1)

            # retrict ranges to between 0 and the end of the signal
            peak_range = peak_range[(peak_range >= 0) & (peak_range <= len(amp))]

            ranges.append(peak_range)

        return ranges

    def get_amps_inds(self, intensity: pt.Series, time_idx: pt.Series) -> tuple:
        # Get the amplitudes and the indices of each peak
        peak_maxima_sign = np.sign(intensity[time_idx])
        peak_maxima_pos = peak_maxima_sign > 0
        peak_maxima_neg = peak_maxima_sign < 0

        if not peak_maxima_sign.dtype == float:
            raise TypeError(
                f"peak_maximas_sign must be float, got {peak_maxima_sign.dtype}"
            )
        if not peak_maxima_pos.dtype == bool:
            raise TypeError(
                f"peak_maxima_pos must be bool, got {peak_maxima_pos.dtype}"
            )
        if not peak_maxima_neg.dtype == bool:
            raise TypeError(
                f"peak_maxima_pos must be bool, got {peak_maxima_neg.dtype}"
            )

        return peak_maxima_sign, peak_maxima_pos, peak_maxima_neg

    def compute_peak_time_ranges(
        self,
        norm_int: FloatArray,
        left: FloatArray,
        right: FloatArray,
        buffer: int,
    ) -> list[IntArray]:
        """
        calculate the range of each peak, returned as a list of ranges. Essentially translates the calculated widths to time intervals, modified by the buffer. Background ranges are defined implicitely as the regions of the time idex not covered by any range.
        """

        norm_int = np.asarray(norm_int, dtype=np.float64)
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)

        if len(norm_int) == 0:
            raise ValueError("amplitude array has length 0")
        if len(left) == 0:
            raise ValueError("left index array has length 0")
        if len(right) == 0:
            raise ValueError("right index array has length 0")

        ranges = self.compute_individual_peak_ranges(
            norm_int,
            left,
            right,
            buffer,
        )

        # Identiy subset ranges
        ranges_mask = self.mask_subset_ranges(ranges)

        # Keep only valid ranges and baselines

        validated_ranges = []
        for i, r in enumerate(ranges):
            if ranges_mask[i]:
                validated_ranges.append(r)

        if len(validated_ranges) == 0:
            raise ValueError(
                "Something has gone wrong with the ranges or the validation"
            )

        return validated_ranges

    # @pa.check_types
    def window_df_factory(
        self,
        time: FloatArray,
        amp: FloatArray,
        left_indices: FloatArray,
        right_indices: FloatArray,
        buffer: int = 0,
    ) -> pt.DataFrame[OutWindowDF_Base]:
        """ """

        time = np.asarray(time, np.float64)
        amp = np.asarray(amp, np.float64)

        left_indices = np.asarray(left_indices, np.float64)
        right_indices = np.asarray(right_indices, np.float64)

        if len(amp) == 0:
            raise ValueError("amplitude array has length 0")
        if len(left_indices) == 0:
            raise ValueError("left index array has length 0")
        if len(right_indices) == 0:
            raise ValueError("right index array has length 0")

        peak_windows: list[IntArray] = self.compute_peak_time_ranges(
            amp, left_indices, right_indices, buffer
        )

        window_df = self._label_windows(
            time,
            peak_windows,
        )

        return window_df

    def _label_windows(
        self, time: FloatArray, peak_windows: list[IntArray]
    ) -> pt.DataFrame[OutWindowDF_Base]:
        # generate a time idx array for indexing the window labels
        time_idx = np.arange(0, len(time), 1)

        # create a frame of window_idx and time_idx for the peaks, with peak_type labeled
        # frame is created from a list of arrays with jagged length, first as rows then
        # transposed, melted and NA's are dropped.
        window_df = (
            pd.DataFrame(peak_windows, dtype=pd.Int64Dtype())
            .T
            # pw_idx: peak_window_idx
            .melt(var_name="pw_idx", value_name="time_idx")
            .dropna()
            .assign(**{"pw_idx": lambda df: (df["pw_idx"] + 1)})
            # here we broadcast the peak windows to the whole time range
            .set_index("time_idx")
            .sort_index()
            .reindex(time_idx)
            .reset_index()
            .astype(
                {
                    "time_idx": pd.Int64Dtype(),
                    "pw_idx": pd.Int64Dtype(),
                }
            )
        )
        # create a labelled series for the discontinuous NA locations. In this context
        # 0 represents peak windows,
        iw_idx, _ = label(window_df["pw_idx"].isna())  # type: ignore

        window_df = (
            window_df.assign(iw_idx=iw_idx)
            # replace the peak window value 0 with the window_idx values already defined
            .assign(
                window_idx=lambda df: df["iw_idx"].mask(df["iw_idx"] == 0, df["pw_idx"])
            )
            # label the peak and interpeak regions
            .assign(window_type="interpeak")
            .astype({"window_type": pd.StringDtype()})
            .assign(
                window_type=lambda df: df["window_type"].mask(df["iw_idx"] == 0, "peak")
            )
            # remove intermediate columns
            .drop(["pw_idx", "iw_idx"], axis=1)
        )
        # need to assign a super window idx which labels each window in time_idx order irrespective of window type

        # get the first time idx value of each window as an aggregate table then label with
        # a cumulatively increasing idx column. 'sw_idx': 'super_window_idx'

        sw_idx_df = (
            window_df.groupby(["window_type", "window_idx"])["time_idx"]
            .agg("first")
            .sort_values()
            .to_frame()
            .assign(**{"sw_idx": 1})
            .assign(
                **{"sw_idx": lambda df: df["sw_idx"].cumsum().astype(pd.Int64Dtype())}
            )
            .reset_index()
            .set_index("time_idx")
            .loc[:, ["sw_idx"]]
        )

        # join the super_window_idx to window_df and foreward fill with the idx value
        # to label each row
        window_df = (
            window_df.set_index("time_idx")
            .join(sw_idx_df, how="left")
            .assign(**{"sw_idx": lambda df: df["sw_idx"].ffill()})
            .reset_index()
        )

        return typing.cast(pt.DataFrame[OutWindowDF_Base], window_df)

    @pa.check_types
    def window_df_pivot(
        self,
        window_df: pt.DataFrame[OutWindowDF_Base],
        time_col: str = "time_idx",
        window_idx_col: str = "window_idx",
        aggfuncs: list = ["min", "max"],
    ) -> pd.DataFrame:
        """
        pivot the window df to neatly provide a description of each window

        TODO:

        - [ ] establish schema of the pivot table
        """

        pivot_window_df = (
            window_df.pivot_table(
                values=time_col,
                columns=window_idx_col,
                aggfunc=aggfuncs,
            )
            .stack(1)
            .reset_index()
        )

        return pivot_window_df

    def display_windows(
        self,
        peak_df: pt.DataFrame[PeakMap],
        signal_df: pt.DataFrame[OutSignalDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
        time_col: str = "time_idx",
        y_col: str = "amp_corrected",
        ax=None,
    ):
        if not ax:
            fig, _ax = plt.subplots(1)
        else:
            _ax = ax

        peak_signal_join = (
            peak_df.set_index(time_col)
            .join(signal_df.set_index(time_col), how="left", validate="1:1")
            .reset_index()
        )

        # the signal

        _ax.plot(signal_df[time_col], signal_df[y_col], label="signal")

        pwtable = self.window_df_pivot(window_df)

        def signal_window_overlay(
            ax,
            signal_df: pt.DataFrame[OutSignalDF_Base],
            pwtable: pd.DataFrame,
            window_idx_col: str = "window_idx",
            y_col: str = "amp_corrected",
        ) -> None:
            """
            Create an overlay of the signal and the windows.
            """

            set2 = mpl.colormaps["Set2"].resampled(
                pwtable.groupby(window_idx_col).ngroups
            )

            for id, window in pwtable.groupby(window_idx_col):
                anchor_x = window["min"].values[0]
                anchor_y = 0
                width = window["max"].values[0] - window["min"].values[0]
                max_height = signal_df[y_col].max()

                rt = Rectangle(
                    xy=(anchor_x, anchor_y),
                    width=width,
                    height=max_height,
                    color=set2.colors[int(id) - 1],  # type: ignore
                )

                ax.add_patch(rt)

            return ax

        signal_window_overlay(_ax, signal_df, pwtable)

        # the peaks

        _ax.scatter(
            peak_signal_join[time_col],
            peak_signal_join[y_col],
            label="peaks",
            color="red",
        )

        # now draw the interpolations determining the peak width
        if not ax:
            fig.show()  # type: ignore
            plt.show()
        else:
            return _ax
