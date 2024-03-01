from numpy.typing import ArrayLike
import holoviews as hv
import hvplot.pandas
import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series
from hplc_py.deconvolution import opt_params

import hplc_py.map_peaks.definitions as mp_defs
from hplc_py.baseline_correction import definitions as bcorr_defs
from hplc_py.baseline_correction.baseline_correction import BaselineCorrection
from hplc_py.common import common_schemas as com_schs
from hplc_py.common import definitions as com_defs
from hplc_py.common.common import compute_timestep, prepare_signal_store
from hplc_py.map_peaks import schemas as mp_schs
from hplc_py.map_peaks.map_peaks import MapPeaks
from hplc_py.map_peaks.viz import PeakMapPlotHandler
from hplc_py.map_windows.map_windows import MapWindows
from hplc_py.map_windows import definitions as mw_defs
from hplc_py.map_windows import schemas as mw_schs
from hplc_py.map_peaks.map_peaks import mp_viz
from hplc_py.deconvolution.deconvolution import PeakDeconvolver
from hplc_py.deconvolution.opt_params import DataPrepper
from hplc_py.deconvolution import opt_params
from hplc_py.deconvolution import schemas as dc_schs
from hplc_py.deconvolution.fit_assessment import calc_fit_scores
from hplc_py.pipeline.deconvolution import deconv as deconv_pipeline
from hplc_py.deconvolution import definitions as dc_defs
from hplc_py.common import caching
from cachier import cachier

from dataclasses import dataclass

from hplc_py.pipeline.hplc_result_analyzer import HPLCResultsAnalzyer


class DeconvolutionPipeline:
    """
    The deconvolution pipeline class, executing the pipeline and storing the state information of the pipeline.

    Will likely integrate the dashboard into this class.

    TODO:
    - add dashboard functionality
    """

    def __init__(self):

        self._X_key = com_defs.X

    def run(
        self,
        data: pd.DataFrame,
        key_time: str = com_defs.KEY_TIME,
        key_amp: str = com_defs.KEY_AMP,
        correct_baseline: bool = True,
        find_peaks_kwargs: mp_defs.MapPeaksKwargs = mp_defs.MapPeaksKwargs(
            wlen=None,
            prominence=0.01,
        ),
        bcorr_kwargs: bcorr_defs.BlineKwargs = bcorr_defs.BlineKwargs(
            n_iter=250,
            window_size=5,
            verbose=True,
        ),
    ):
        """
        Take the dataset and return all of the results
        """

        # set stateful attributes

        # the key referring to the 'active' signal column, the one we want to input into the pipeline
        # downstream of 'preprocess'. For example, if `correct_baseline` is True, then this is set to
        # "corrected", else its set to the input `key_amp` value.

        # internally defined access keys. use the mapping below if we desire to swap between

        self.X = com_defs.X

        self.key_input_time = key_time
        self.key_input_amp = key_amp

        self.X_idx = com_defs.X_IDX
        self.X_corrected = com_defs.X + "_corrected"

        # calculate timestep

        def calculate_timestep(time: ArrayLike):
            timestep = np.mean(np.diff(time))
            return timestep

        self._timestep = calculate_timestep(data[self.key_input_time])

        # store the mapping between the internal keys and input keys for each phenomenon
        self._key_mappings = {
            com_defs.KEY_TIME: key_amp,
            com_defs.X: key_time,
        }

        self.store_signals: pd.DataFrame = self.pipe_preprocess_data(
            data=data,
            key_time=key_time,
            key_amp=key_amp,
            correct_baseline=correct_baseline,
            bcorr_kwargs=bcorr_kwargs,
        )
        if correct_baseline:
            self.active_signal = self.X_corrected
        else:
            self.active_signal = self.X

        # self.get_hplc_results()

        # as pandera schemas do not handle variable column names well, need to keep a 'X' data structure that for X is either the raw or preprocessed signal. An option would be to pass the raw through a preprocessor, and label it X regardless, and keep the raw signal seperate.As I expect to add preprocessing downstream, this is an acceptable solution. baseline correction etc. falls into the class of 'preprocessing'.

        # if `correct_baseline` == True, correct input amplitude, add results to signal_storage and set the active `X_key` to the 'corrected' value.

        peak_map: dict
        peak_map_plot_handler: Any
        peak_map, peak_map_plot_handler = self.pipe_map_peaks(
            signal_store=self.store_signals,
            find_peaks_kwargs=find_peaks_kwargs,
        )

        self.maxima: pd.DataFrame = peak_map["maxima"]
        self.widths = peak_map["widths"]
        self.contour_line_bounds = peak_map["contour_line_bounds"]

        # map windows
        self.X_w, peak_window_spans = self.pipe_window_X(
            store_signals=self.store_signals,
            contour_line_bounds=peak_map["contour_line_bounds"],
        )
        ##########################################################
        # prepare normalised peak : window tbl and window time / index bounds table

        from types import SimpleNamespace

        # normalized or otherwise core tables primarily for join operations (?)
        self.tbls = SimpleNamespace()

        self.tbls.peak_idx_X_idx = (
            peak_map["maxima"]
            .pipe(pl.from_pandas)
            .filter(pl.col("dim").is_in(["X_idx"]))
            .pivot(index="p_idx", columns="dim", values="value")
            .select(pl.col("p_idx"), pl.col("X_idx").cast(int).cast(str))
            .to_pandas()
        )

        # mapping window type to index to X idx
        self.tbls.window_to_X_idx = (
            self.X_w.pipe(pl.from_pandas)
            .select(
                pl.col("w_type"),
                pl.col("w_idx"),
                pl.col("X_idx"),
            )
            .to_pandas()
        )

        # mapping the windows to the peaks

        self.tbls.window_to_peak_idx = (
            self.tbls.peak_idx_X_idx.pipe(pl.from_pandas)
            .join(
                self.tbls.window_to_X_idx.pipe(pl.from_pandas).with_columns(
                    pl.col("X_idx").cast(str)
                ),
                on="X_idx",
            )
            .to_pandas()
        )

        # TODO: reorganise all data into tbls of 'p_idx','param','dim','unit','value', in that order. that precludes 'idx' as a unit, and rather an id, since the x ('mins') and y ('amp') is both accessed by an idx.

        self.maxima: pl.DataFrame = (
            self.maxima.pipe(pl.from_pandas)
            .pivot(index=["p_idx", "param"], columns="dim", values="value")
            .rename({
                'X_idx':'idx',
                'x':'mins',
            })
            .melt(id_vars=['p_idx','param'], value_vars=['idx','mins','X'], variable_name='unit', value_name='value')
            .with_columns(
                pl.when(pl.col('unit').is_in(['idx','mins'])).then(pl.lit('x')).when(pl.col('unit')=='X').then(pl.lit('y')).alias('dim')
            )
            .pivot(index=['p_idx','param'], columns='unit', values='value')
            .with_columns(pl.col('idx').cast(int))
            .melt(
                id_vars=['p_idx','idx','param'], value_vars=['mins','X'],variable_name='unit',value_name='value'
            )
            .rename({"param":"msnt"})
        )  # fmt: skip

        self.widths = (
            self.widths.pipe(pl.from_pandas)
            .pivot(index=["p_idx", "msnt"], columns="unit", values="value")
            .with_columns(
                pl.col("X_idx").cast(int).alias("idx"),
            )
            .select(
                pl.col("p_idx"),
                pl.col("idx"),
                pl.col("msnt"),
                pl.col("x").alias("mins"),
            )
            .melt(
                id_vars=["p_idx", "idx", "msnt"],
                value_vars="mins",
                variable_name="unit",
                value_name="value",
            )
        )

        self.contour_line_bounds = (
            self.contour_line_bounds.pipe(pl.from_pandas)
            .rename({"loc": "side"})
            .pivot(index=["p_idx", "side", "msnt"], columns="dim", values="value")
            .select(
                pl.col(["p_idx", "side", "msnt"]),
                pl.col("X_idx_rounded").cast(int).alias("idx"),
                pl.col("x").alias("mins"),
                pl.col("X"),
            )
            .melt(
                id_vars=["p_idx", "idx", "side", "msnt"],
                value_vars=["mins", "X"],
                variable_name="unit",
                value_name="value",
            )
        )

        # join all the tables back together, with a null 'side' column in widths and maxima to conform to the contour_line_bounds tbl shape.
        self.tbls.peak_map = pl.concat(
            [
                self.maxima.with_columns(
                    pl.col(["p_idx", "idx"]),
                    pl.lit(None).alias("side"),
                    pl.col(["msnt", "unit", "value"]),
                ),
                self.widths.with_columns(
                    pl.col(["p_idx", "idx"]),
                    pl.lit(None).alias("side"),
                    pl.col(["msnt", "unit", "value"]),
                ),
                self.contour_line_bounds,
            ],
            how="diagonal_relaxed",
        ).select("p_idx", "idx", "msnt", "side", "unit", "value")

        # now window it.

        self.tbls.peak_map = (
            self.tbls.window_to_peak_idx.pipe(pl.from_pandas)
            .select("w_type", "w_idx", "p_idx")
            .join(self.tbls.peak_map, on="p_idx", how="left")
        )

        # now prepare 'peak_msnts_windowed' tbl to feed into dataprepper_pipeline.
        # need the maxima X_idx and X, width_whh in X_idx labelled 'width' in 'dim' column.
        peak_msnts_windowed = (
            self.tbls.peak_map.filter(pl.col("msnt").is_in(["maxima", "width_whh"]))
            .select(
                pl.col(["w_type", "w_idx", "p_idx", "msnt"]),
                pl.when(pl.col("msnt") == "width_whh")
                .then(pl.lit("width"))
                .when(pl.col("unit") == "mins")
                .then(pl.lit("X_idx"))
                .otherwise(pl.col("unit"))
                .alias("dim"),
                pl.when(pl.col("unit") == "mins")
                .then(pl.col("value").truediv(self._timestep))
                .otherwise(pl.col("value"))
                .alias("value"),
            )
            .with_columns(pl.col("msnt").replace({"width_whh": "whh"}))
            .to_pandas()
            .pipe(dc_schs.PeakMsntsWindowed.validate, lazy=True)
        )

        # deconvolution

        # peak_msnts_windowed - a table of p_idx, loc, dim, and value columns.
        params: DataFrame[dc_schs.Params] = self.dataprepper_pipeline(
            peak_msnts_windowed=peak_msnts_windowed,
            X_windowed=self.X_w,
            timestep=self._timestep,
        )
        breakpoint()
        deconv_results: dc_schs.DeconvolutionOutput = self.pipe_deconvolution(
            X_windowed=self.X_w, params=params
        )

        self.prepare_peak_map_with_params_and_popts(
            peak_msnts_windowed=peak_msnts_windowed,
            params=params,
            popt=deconv_results.popt,
        )

        fit_report = calc_fit_scores(X_w_with_recon=deconv_results.X_w_with_recon)

        self.plot_results(
            X_w_with_recon=deconv_results.X_w_with_recon,
            p_signals=deconv_results.psignals,
        )

        breakpoint()
        return None

    def dataprepper_pipeline(
        self,
        peak_msnts_windowed: DataFrame[dc_schs.PeakMsntsWindowed],
        X_windowed: DataFrame[mw_schs.X_Windowed],
        timestep: float,
    ):

        params: DataFrame[dc_schs.Params] = opt_params.params_factory(
            peak_msnts_windowed=peak_msnts_windowed,
            X_w=X_windowed,
            timestep=self._timestep,
        )
        
        

        breakpoint()
        return params

    def prepare_peak_map_with_params_and_popts(
        self,
        peak_msnts_windowed: DataFrame[dc_schs.PeakMsntsWindowed],
        params: DataFrame[dc_schs.Params],
        popt: DataFrame[dc_schs.Popt],
    ):
        """
        Arrange the various calculations of each peak parameter in one table for easy comparison.
        """

        # join them all together

        # arrange everything to match the params format.

        peak_msnts_windowed_ = (
            peak_msnts_windowed.pipe(pl.from_pandas)
            .select(
                pl.col(dc_defs.W_IDX),
                pl.col(mw_defs.P_IDX),
                pl.col(mp_defs.DIM).replace(
                    {
                        mp_defs.X: mp_defs.MAXIMA,
                        mp_defs.X_IDX: mp_defs.LOC,
                    }
                ),
                pl.col(mp_defs.VALUE),
            )
            .rename({mp_defs.DIM: dc_defs.PARAM, dc_defs.VALUE: "actual"})
        )

        popt_ = (
            popt.pipe(pl.from_pandas)
            .melt(
                id_vars=[
                    dc_defs.W_IDX,
                    dc_defs.P_IDX,
                ],
                value_vars=[dc_defs.MAXIMA, dc_defs.MSNT, dc_defs.WIDTH, dc_defs.SKEW],
                variable_name=dc_defs.PARAM,
                value_name="popt",
            )
            .sort([dc_defs.W_IDX, dc_defs.P_IDX, dc_defs.PARAM])
        )

        report_tbl = (
            params.pipe(pl.from_pandas)
            .with_columns(pl.col(dc_defs.PARAM).cast(str))
            .join(popt_, on=[dc_defs.P_IDX, dc_defs.PARAM, dc_defs.W_IDX], how="left")
            .join(
                peak_msnts_windowed_,
                on=[dc_defs.P_IDX, dc_defs.PARAM, dc_defs.W_IDX],
                how="left",
            )
            .select(
                pl.col(
                    [
                        dc_defs.W_TYPE,
                        dc_defs.W_IDX,
                        dc_defs.P_IDX,
                        dc_defs.PARAM,
                        dc_defs.ACTUAL,
                        dc_defs.KEY_P0,
                        dc_defs.KEY_LB,
                        dc_defs.KEY_UB,
                        dc_defs.KEY_POPT,
                    ]
                )
            )
        )

        return report_tbl

    def plot_results(
        self,
        X_w_with_recon: DataFrame[dc_schs.X_Windowed_With_Recon],
        p_signals: DataFrame[dc_schs.PSignals],
    ):

        X_plot = X_w_with_recon.pipe(pl.from_pandas).plot(x="X_idx", y=["X"])

        recon_plot = X_w_with_recon.pipe(pl.from_pandas).plot(x="X_idx", y=["recon"])

        p_signals_plot = p_signals.pipe(pl.from_pandas).plot.area(
            x="X_idx", y="unmixed", group_by="p_idx"
        )

        hvplot.show(p_signals_plot)

    def get_hplc_results(
        self,
    ):
        """
        Run the original hplc_py `fit_peaks` to compare the results.
        """

        self.cremerlab_results = HPLCResultsAnalzyer(
            data=self._input_data, cols={"time": "x", "signal": "y"}
        )

    def pipe_deconvolution(
        self,
        X_windowed: DataFrame[mw_schs.X_Windowed],
        params: DataFrame[dc_schs.Params],
    ) -> dc_schs.DeconvolutionOutput:
        """
        From peak map we require the peak location, maxima, and WHH. Provide them as a table called 'OptParamPeakInput'. Provide it in long form.
        """

        # deconvolve

        peak_deconv = PeakDeconvolver()
        peak_deconv.fit(X_w=X_windowed, params=params)

        peak_deconv.transform()

        deconv_output = dc_schs.DeconvolutionOutput(
            popt=peak_deconv.popt,
            psignals=peak_deconv.psignals,
            rsignal=peak_deconv.recon,
            X_w_with_recon=peak_deconv.X_w_with_recon,
        )

        return deconv_output

    @cachier(hash_func=caching.custom_param_hasher, cache_dir=caching.CACHE_PATH)
    def pipe_preprocess_data(
        self,
        data: pd.DataFrame,
        key_time: str,
        key_amp: str,
        correct_baseline: bool,
        bcorr_kwargs: bcorr_defs.BlineKwargs = bcorr_defs.BlineKwargs(),
    ) -> pd.DataFrame:
        """
        apply preprocessing as required, returning a preprocessed dataset X, and a dict of artifacts - the time array, and raw array. If no preprocessing is selected, the values of the raw array will equal the values of the X column.
        """
        signal_store: pl.DataFrame = (
            data.pipe(pl.from_pandas)
            .with_row_index(name=self.X_idx)
            .select(
                [
                    pl.col(self.X_idx).cast(int),
                    pl.col(key_time).alias(self.key_input_time),
                    pl.col(key_amp).alias(self.X),
                ]
            )
        )

        correct_baseline_input = (
            signal_store
            .select(
                pl.col('X_idx'),
                pl.col('X')
            )
            .to_pandas()
            .pipe(DataFrame[com_schs.X_Schema])
            
        )
        
        if correct_baseline:
            data_X: pd.DataFrame
            bcorr_plot: Any
            data_X, bcorr_plot = self.pipe_correct_baseline(
                data=correct_baseline_input,
                **bcorr_kwargs,
            )

        X_corrected: pl.DataFrame = (
            data_X.pipe(pl.from_pandas)
            .filter(pl.col("signal") == "corrected")
            .select(pl.col(self.X).alias(self.X_corrected), pl.col(self.X_idx))
        )

        signal_store_: pd.DataFrame = (
            signal_store.join(X_corrected, on=self.X_idx)
            .select(
                [
                    pl.col(self.X_idx),
                    pl.col(self.key_input_time),
                    pl.col(self.X),
                    pl.col(self.X_corrected),
                ]
            )
            .melt(
                id_vars=[self.X_idx, self.key_input_time],
                variable_name="signal",
                value_name="amplitude",
            )
            .to_pandas()
        )

        return signal_store_

    def pipe_window_X(
        self,
        store_signals,
        contour_line_bounds: pd.DataFrame,
    ) -> tuple[DataFrame[mw_schs.X_Windowed], Any]:
        """
        Window X based on peak interval overlaps
        """

        left_bases = (
            contour_line_bounds.pipe(pl.from_pandas)
            .filter(
                pl.col(mp_defs.KEY_MSNT) == mp_defs.KEY_PB,
                pl.col(mp_defs.DIM) == mp_defs.KEY_X_IDX_ROUNDED,
                pl.col(mp_defs.LOC) == mp_defs.KEY_LEFT,
            )
            .select(mp_defs.VALUE)
            .to_series()
            .to_pandas()
            .rename("pb_left")
            .pipe(Series[float])
        )

        right_bases: Series[float] = (
            contour_line_bounds.pipe(pl.from_pandas)
            .filter(
                pl.col(mp_defs.KEY_MSNT) == mp_defs.KEY_PB,
                pl.col(mp_defs.DIM) == mp_defs.KEY_X_IDX_ROUNDED,
                pl.col(mp_defs.LOC) == mp_defs.KEY_RIGHT,
            )
            .select(mp_defs.VALUE)
            .to_series()
            .to_pandas()
            .rename("pb_right")
            .pipe(Series[float])
        )

        X_in: DataFrame[com_schs.X_Schema] = (
            store_signals.pipe(pl.from_pandas)
            .filter(pl.col(bcorr_defs.KEY_SIGNAL) == self.active_signal)
            .pivot(index=self.X_idx, columns="signal", values="amplitude")
            .select(pl.col(self.X_idx), pl.col(self.X_corrected).alias(self.X))
            .to_pandas()
            .pipe(DataFrame[com_schs.X_Schema])
        )

        map_windows = MapWindows(X=X_in, left_bases=left_bases, right_bases=right_bases)

        window_spans = map_windows.plot.draw_peak_windows()
        X_windowed = map_windows.X_windowed

        return X_windowed, window_spans

    def pipe_map_peaks(
        self,
        signal_store: DataFrame[com_schs.X_Schema],
        find_peaks_kwargs: mp_defs.MapPeaksKwargs = mp_defs.map_peaks_kwargs_defaults,
        viz_maxima: bool = True,
        viz_whh: bool = False,
        viz_bases: bool = True,
    ) -> tuple[dict, PeakMapPlotHandler]:

        X: DataFrame[com_schs.X_Schema] = (
            signal_store.pipe(pl.from_pandas)
            .filter(pl.col("signal") == self.active_signal)
            .pivot(index=self.X_idx, columns="signal", values="amplitude")
            .select(pl.col(self.X_idx), pl.col(self.X_corrected).alias(self.X))
            .to_pandas()
            .pipe(DataFrame[com_schs.X_Schema])
        )

        map_peaks = MapPeaks(X=X, find_peaks_kwargs=find_peaks_kwargs)
        peak_map = map_peaks.peak_map

        peak_map_plots: mp_viz.PeakMapPlotHandler = map_peaks.plot.draw_peak_mappings(
            signal=False,
            maxima=viz_maxima,
            whh=viz_whh,
            bases=viz_bases,
        )
        # add a time column to each of the peak map tables. Add it via timestep transformation

        contour_line_bounds = (
            peak_map.contour_line_bounds.pipe(pl.from_pandas)
            .pivot(
                index=[
                    "p_idx",
                    "loc",
                    "msnt",
                ],
                columns="dim",
                values="value",
            )
            .with_columns(
                pl.col("X_idx_rounded").mul(self._timestep).alias(self.key_input_time),
            )
            .melt(
                id_vars=["p_idx", "loc", "msnt"],
                variable_name="dim",
                value_name="value",
            )
            .to_pandas()
        )

        maxima = (
            peak_map.maxima.pipe(pl.from_pandas)
            .pivot(index=["p_idx", "loc"], columns=["dim"], values="value")
            .select(
                pl.col("p_idx"),
                pl.col("loc").alias("param"),
                pl.col("X_idx"),
                pl.col("X_idx").mul(self._timestep).alias(self.key_input_time),
                pl.col("X"),
            )
            .melt(id_vars=["p_idx", "param"], variable_name="dim", value_name="value")
            .to_pandas()
        )

        widths = (
            peak_map.widths.pipe(pl.from_pandas)
            .select(
                pl.col("p_idx"),
                pl.col("msnt"),
                pl.col("value").alias("X_idx"),
                pl.col("value").mul(self._timestep).alias(self.key_input_time),
            )
            .melt(id_vars=["p_idx", "msnt"], variable_name="unit", value_name="value")
            .to_pandas()
        )

        peak_map = {
            "contour_line_bounds": contour_line_bounds,
            "maxima": maxima,
            "widths": widths,
        }

        return peak_map, peak_map_plots

    def pipe_correct_baseline(
        self,
        data: DataFrame[com_schs.X_Schema],
        n_iter: int = 250,
        window_size: float = 5,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, Any]:

        bcorr = BaselineCorrection(
            n_iter=n_iter, window_size=window_size, verbose=verbose
        )

        X_np = data.pipe(pl.from_pandas).select(com_defs.X).to_series().to_numpy()

        signals = bcorr.fit(X=X_np).correct_baseline()
        bcorr_plot = bcorr.viz_baseline_correction(show=False)

        return signals, bcorr_plot
