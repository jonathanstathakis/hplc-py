import holoviews as hv

hv.extension("bokeh")
from IPython.core.debugger import set_trace
import pandera as pa
from hplc_py import ROOT
import hvplot
import panel as pn
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from hplc_py.baseline_correction import compute_background
from sklearn.compose import ColumnTransformer
from typing import Self
import polars as pl
from hplc_py.map_signal.map_peaks import map_peaks
from hplc_py.baseline_correction import baseline_correction
from dataclasses import dataclass
import logging
from hplc_py.map_signal.map_windows import map_windows
from hplc_py.map_signal.map_peaks import definitions as mp_defs
from hplc_py.map_signal.map_peaks.definitions import map_peaks_kwargs_defaults
from pandera.typing import DataFrame
from pandera.typing.polars import DataFrame as polars_DataFrame
import pandera.polars as papl

# pn.config.theme = "dark"


def exception_handler(ex):
    logging.error("Error", exc_info=ex)
    pn.state.notifications.error("Error: %s" % ex)


pn.extension(
    sizing_mode="stretch_width",
    exception_handler=exception_handler,
    notifications=True,
    design="bootstrap",
)


class DashBoard:
    def __init__(self):
        # for more info on BootStrap see - <https://panel.holoviz.org/reference/templates/Bootstrap.html>
        self._template = pn.template.BootstrapTemplate(title="Prototype")

        # for more infor on GridBox, see - <https://panel.holoviz.org/reference/layouts/GridBox.html>
        self._template.main.append(pn.GridBox(nrows=2, ncols=2))
        self.grid = self._template.main[0]

        for i in range(4):
            self.grid.append(pn.Column(f"sector {i}"))

        self._template.servable()

    def add_bound_bcorr(self, bcorr_func, n_iter, **other_kwargs):
        """
        bcorr_func: a function that takes n_iter and returns a viz object
        """

        bcorr_slider = pn.widgets.IntSlider(
            value=n_iter, start=1, end=300, name="n_iter"
        )
        interactive_bcorr = pn.bind(bcorr_func, n_iter=bcorr_slider, **other_kwargs)

        self.grid[0] = pn.Column(bcorr_slider, interactive_bcorr, scroll=True)

    def add_peak_map_plot(self, peak_map_func, find_peaks_kwargs):
        """
        peak_map func, a function that takes an input signal and kwargs and outputs a peak map viz
        """

        peak_map_viz = peak_map_func(find_peaks_kwargs=find_peaks_kwargs)

        self.grid[1] = pn.Column(peak_map_viz)


class Signal(papl.DataFrameModel):
    idx: int
    time: float
    signal: float
    adjusted_signal: float
    background: float
    window_type: str
    window_idx: int

    class Config:
        strict = True
        ordered = True
        coerce = True
        name = "Signal"


class Windows(papl.DataFrameModel):
    w_type: str
    w_idx: int


class PreProcesser:
    def __init__(self):

        self.ct = get_column_transformer()
        self._input_time_col = ""
        self._input_amp_col = ""

        # define the signal storage
        @dataclass(frozen=True, init=False)
        class SignalKeys:
            idx: str = "idx"
            time: str = "time"
            signal: str = "signal"
            adjusted_signal: str = "adjusted_signal"
            background: str = "background"
            window_type: str = "w_type"
            window_idx: str = "w_idx"

        self.signal_keys = SignalKeys()

        self._signal_schema = {
            self.signal_keys.idx: pl.Int64,
            self.signal_keys.time: pl.Float64,
            self.signal_keys.signal: pl.Float64,
            self.signal_keys.adjusted_signal: pl.Float64,
            self.signal_keys.background: pl.Float64,
            self.signal_keys.window_type: pl.String,
            self.signal_keys.window_idx: pl.Int64,
        }

        self.signal: polars_DataFrame[Signal] = polars_DataFrame[Signal](
            schema=self._signal_schema
        )

        # status flags, the processes need to be run in order
        self.signal_injested = False
        self.signal_adjusted = False
        self.peaks_mapped = False
        self.windows_mapped = False

    def ingest_signal(self, signal: pl.DataFrame, time_col, amp_col):

        if not isinstance(signal, pl.DataFrame):
            raise TypeError("please input polars df")

        self._input_time_col = time_col
        self._input_amp_col = amp_col

        self._in_signal = signal

        # construct the signal store from the input time and signal columns, add the remaining expected columns to the right

        inter_df = pl.DataFrame({"idx": list(range(len(signal)))})

        inter_df = inter_df.with_columns(
            signal.select(time_col)
            .to_series(0)
            .cast(float)
            .alias(self.signal_keys.time),
            signal.select(amp_col)
            .to_series(0)
            .cast(float)
            .alias(self.signal_keys.signal),
        )

        self.signal = self.signal.join(
            inter_df,
            on=[self.signal_keys.idx, self.signal_keys.time, self.signal_keys.signal],
            how="outer_coalesce",
        )

        self.signal_injested = True

    def signal_adjustment(
        self,
        bcorr__n_iter=50,
    ):
        self.bcorr__n_iter = bcorr__n_iter
        """
        For all signal preprocessing prior to peak mapping and windowing
        """

        if not self.signal_injested:
            raise RuntimeError("Run 'ingest_signal' before 'preprocess")

        self.ct.set_params(bcorr__n_iter=bcorr__n_iter).set_output(transform="polars")

        out: pl.DataFrame = self.ct.fit_transform(self.signal)

        # ct.fit_transform returns 1 column

        # add adjusted signal to central data store
        self.signal.replace(self.signal_keys.adjusted_signal, out.to_series(0))

        # add background to central data store
        background = self.ct.named_transformers_["bcorr"].background_

        self.signal.replace(self.signal_keys.background, background)

        self.signal_adjusted = True

        return self.signal

    def map_peaks(
        self,
        find_peaks_kwargs: mp_defs.MapPeaksKwargs = {"wlen": None, "prominence": 0.01},
    ):
        """
        Find the peak dimensions. Run after 'signal_adjustment'
        """

        if not self.signal_adjusted:
            raise RuntimeError("Run signal_adjustment first")

        map_peaks_in: pd.DataFrame = self.signal.select(
            pl.col("idx").cast(int), pl.col(self.signal_keys.adjusted_signal).alias("X")
        ).to_pandas()

        self.peak_mapper = map_peaks.PeakMapper(find_peaks_kwargs=find_peaks_kwargs)
        self.peak_map = self.peak_mapper.fit_transform(X=map_peaks_in)

        viz_maxima = True
        viz_whh = True
        viz_bases = "pb"

        pm_viz: map_peaks.mp_viz.PeakMapViz = self.peak_mapper.viz(
            maxima=viz_maxima,
            whh=viz_whh,
            base=viz_bases,
        )

        peak_base_viz = pm_viz.bases

        self.peaks_mapped = True

        return peak_base_viz

    def map_windows(self):

        if not self.peaks_mapped:
            raise RuntimeError("run map_peaks first")

        left_bases = self.get_base_side(
            df=self.peak_map.contour_line_bounds,
            side="left",
            msnt="pb",
            unit="idx_rounded",
        )
        right_bases = self.get_base_side(
            df=self.peak_map.contour_line_bounds,
            side="right",
            msnt="pb",
            unit="idx_rounded",
        )

        self.window_mapper = map_windows.MapWindows(
            left_bases=left_bases, right_bases=right_bases
        )

        self.window_mapper.fit_transform(
            X=self.signal.select(
                pl.col("idx").cast(int), pl.col("adjusted_signal").alias("X")
            ).to_pandas()
        )

        self.add_windows_to_signal(
            windows=self.window_mapper.X_windowed_[["w_type", "w_idx"]].pipe(
                pl.from_pandas
            )
        )

        self.windows_mapped = True

    @pa.check_types
    def add_windows_to_signal(self, windows: DataFrame[Windows]):
        """"""

        for col in ["w_type", "w_idx"]:
            self.signal = self.signal.replace(col, windows.select(col).to_series())

    def get_base_side(self, df, side, msnt, unit):

        base = (
            df.pipe(pl.from_pandas)
            .filter(
                pl.col("msnt") == msnt,
                pl.col("dim") == unit,
                pl.col("loc") == side,
            )
            .select("value")
            .to_series()
            .rename(f"{msnt}_{side}")
        )
        return base

    def viz_preprocessing(self, opt_args={}) -> hv.core.overlay.Overlay:
        """
        overlay the background, corrected signal, peak mappings and window mappings to
        provide a viz of the result
        """

        def interpeak_window_spans(interpeak_window_bounds: pl.DataFrame):
            start_idx = interpeak_window_bounds.columns.index("start")
            end_idx = interpeak_window_bounds.columns.index("end")

            x_start = interpeak_window_bounds.to_series(start_idx)
            x_end = interpeak_window_bounds.to_series(end_idx)

            interpeak_spans = hv.VSpans((x_start, x_end), label="interpeak wdw").opts(
                alpha=0.1,
            )

            return interpeak_spans

        signal_viz = self.signal.plot(
            x="idx",
            y=["adjusted_signal", "background"],
            line_alpha=[1.0, 0.5],
            line_dash=["solid", "dotdash"],
        )

        interpeak_window_bounds = self.window_mapper.window_bounds_.filter(
            pl.col("w_type").eq("interpeak")
        )
        interpeak_spans = interpeak_window_spans(
            interpeak_window_bounds=interpeak_window_bounds
        )

        pm_viz = self.peak_mapper.viz(whh=False).overlay()

        def get_window_label_coords(df: pl.DataFrame):
            """
            create a label object labeling each window center top. Use data coordinates, x needs to be the middle of each window, y needs to be maxima + 20.

            Pleae input the Signal table from `initial_results.signals`
            """
            # get x coords as the mean of each windows x values
            window_label_x_coords = (
                df.filter(pl.col("w_type").eq("peak"))
                .select(pl.col("idx", "adjusted_signal", "w_idx"))
                .group_by(["w_idx"], maintain_order=True)
                .agg(pl.col("idx").mean().alias("x"))
            )

            # get the y coord as the maxima of each `n_iter` signal

            window_label_y_coord = df.select(
                pl.col("adjusted_signal").max().mul(1.05).alias("y")
            )

            assert (
                len(window_label_y_coord) == 1
            ), "expect a length of 1, the global maxima of the signal"

            # join them so that each [n_iter, window] has an x and y

            window_label_coords = window_label_x_coords.join(
                window_label_y_coord, how="cross"
            )

            return window_label_coords

        window_label_coords = self.signal.pipe(get_window_label_coords)

        window_labels = window_label_coords.plot.labels(x="x", y="y", text="w_idx")

        out_viz = pm_viz * signal_viz * interpeak_spans * window_labels

        return out_viz


def get_bcorr_tformer():
    bcorr = baseline_correction.SNIPBCorr()
    return bcorr


def get_column_transformer():

    bcorr = get_bcorr_tformer()

    transformers = [("bcorr", bcorr, ["signal"])]
    ct = ColumnTransformer(transformers=transformers)

    return ct


def get_bcorr_viz(n_iter, prepro_obj: PreProcesser, height: int, width: int):

    prepro_obj.signal_adjustment(bcorr__n_iter=n_iter)
    viz = prepro_obj.ct.named_transformers_["bcorr"].viz(height=height, width=width)

    return viz


def main():

    prepro = PreProcesser()
    prepro.ingest_signal(ringland_dset(), time_col="time", amp_col="signal")

    dboard = DashBoard()

    bcorr_kwargs = dict(height=500, width=500)
    try:
        dboard.add_bound_bcorr(
            get_bcorr_viz, n_iter=40, prepro_obj=prepro, **bcorr_kwargs
        )
    except Exception as e:
        # breakpoint()
        raise

    find_peaks_kwargs = {}

    dboard.add_peak_map_plot(prepro.map_peaks, find_peaks_kwargs=find_peaks_kwargs)


if __name__ == "__main__":
    main()

if __name__.startswith("bokeh"):
    # start with panel serve script.py
    main()
