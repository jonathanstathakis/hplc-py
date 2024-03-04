from types import SimpleNamespace
from hplc_py.deconvolution import deconvolution
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

        self.__W_TYPE = "w_type"
        self.__W_IDX = "w_idx"
        self.__X_IDX = "X_idx"
        self.__AMPLITUDE = "amplitude"
        self.__UNIT_IDX = "unit_idx"
        self.__TIME_UNIT = "x"
        self.__SIGNAL = "signal"
        self.__MAXIMA = "maxima"
        self.__DIM = "dim"
        self.__P_IDX = "p_idx"
        self.__VALUE = "value"
        self.__PARAM = "param"
        self.__IDX = "idx"
        self.__UNIT = "unit"
        self.__Y = "y"
        self.__X = "X"
        self.__LOC = "loc"
        self.__SIDE = "side"
        self.__MSNT = "msnt"
        self.__X_IDX_ROUNDED = "X_idx_rounded"
        self.__UNASSIGNED = "unassigned"
        self.__WIDTH = "width"
        self.__POINT = "point"
        self.__CONTOUR_LINE_BOUND = "contour_line_bound"
        self.__TYPE = "type"
        self.__RECON = "recon"
        self.__X_CORRECTED = "X_corrected"
        self.__LOCATION = "location"
        self.__SCALE = "scale"
        self.__SKEW = "skew"

        self.__WHH = "whh"

        self.__CONTOUR_LINE_BOUNDS = "contour_line_bounds"
        self.__WIDTHS = "widths"

    def __eq__(self, other):
        """
        X                                       active_signal                           fit_report                              maxima                                  pipe_preprocess_data                    run
        X_corrected                             contour_line_bounds                     get_hplc_results                        pipe_correct_baseline                   pipe_window_X                           store_signals
        X_idx                                   cremerlab_results                       key_input_amp                           pipe_deconvolution                      plot_results                            tbls
        X_w                                     dataprepper_pipeline                    key_input_time                          pipe_map_peaks                          prepare_peak_map_with_params_and_popts  widths
        """

        # for df in ["store_signals",""]

        return True

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
        debug: bool = True,
    ):
        """
        Take the dataset and return all of the results
        """

        # set constants
        # set stateful attributes

        # the key referring to the 'active' signal column, the one we want to input into the pipeline
        # downstream of 'preprocess'. For example, if `correct_baseline` is True, then this is set to
        # "corrected", else its set to the input `key_amp` value.

        self.input_keys_time = key_time
        self.input_keys_amp = key_amp

        # internally defined access keys. use the mapping below if we desire to swap between

        self.keys_X = com_defs.X
        self.keys_X_idx = com_defs.X_IDX
        self.keys_X_corrected = com_defs.X + "_corrected"

        # which_x_unit = "unit_idx"
        which_x_unit = self.input_keys_time
        if which_x_unit == "unit_idx":
            x_unit = "unit_idx"
        elif which_x_unit == self.input_keys_time:
            x_unit = self.input_keys_time
        else:
            raise ValueError("please input one of 'unit_idx', self.input_keys_time")

        self._input_data = data

        if debug:
            self.get_hplc_results()

        # calculate timestep

        def calculate_timestep(time: ArrayLike):
            timestep = np.mean(np.diff(time))
            return timestep

        self._timestep = calculate_timestep(data[self.input_keys_time])

        self.tbl_signals: pd.DataFrame = self.pipe_preprocess_data(
            data=data,
            key_time=self.input_keys_time,
            key_amp=self.input_keys_amp,
            correct_baseline=correct_baseline,
            bcorr_kwargs=bcorr_kwargs,
        )

        # if `correct_baseline` == True, correct input amplitude, add results to signal_storage and set the active `X_key` to the 'corrected' value.

        if correct_baseline:
            self.keys_active_signal = self.keys_X_corrected
        else:
            self.keys_active_signal = self.keys_X

        map_peaks_output: dict
        peak_map_plot_handler: Any
        map_peaks_output, peak_map_plot_handler = self.pipe_map_peaks(
            signal_store=self.tbl_signals,
            find_peaks_kwargs=find_peaks_kwargs,
        )

        # map windows
        self.X_w, peak_window_spans = self.pipe_window_X(
            store_signals=self.tbl_signals,
            contour_line_bounds=map_peaks_output["contour_line_bounds"],
        )
        ##########################################################
        # prepare normalised peak : window tbl and window time / index bounds table

        self.update_signal_tbl_with_windows()

        # normalized or otherwise core tables primarily for join operations (?)

        self.prepare_tables(map_peaks_output)

        # deconvolution

        # creating the "X_windowed" input from the signal store based on the selected `x_unit`

        X_windowed_input = (
            self.tbl_signals.pipe(pl.from_pandas)
            .filter(pl.col(self.__SIGNAL).eq(self.keys_active_signal))
            .select([self.__W_TYPE, self.__W_IDX, x_unit, self.__AMPLITUDE])
            .to_pandas()
        )

        params: DataFrame[dc_schs.Params] = self.curve_fit_input_pipeline(
            peak_map=self.tbl_peak_map,
            X_windowed=X_windowed_input,
            x_unit=x_unit,
        )

        self.tbl_popt = self.pipe_deconvolution(
            X_windowed=X_windowed_input, params=params, x_unit=x_unit
        ).pipe(pl.from_pandas)

        peak_signals = deconvolution.construct_peak_signals(
            X_w=X_windowed_input, popt=self.tbl_popt.to_pandas(), x_unit=x_unit
        )

        recon_df = deconvolution.reconstruct_signal(
            peak_signals=peak_signals, x_unit=x_unit
        )

        # add recon to signal store

        self.update_signal_tbl_with_recon(recon_df)

        self.pipe_fit_scores(x_unit)

        # self.plot_results(
        #     X_w_with_recon=deconv_results.X_w_with_recon,
        #     p_signals=deconv_results.psignals,
        # )

        # !! DEBUGGING

        # !! input hplc params into my deconvolution pipeline and see how the results differ

        self.pipe_results_comparison_with_cremerlab(params)

        breakpoint()
        return None

    def pipe_results_comparison_with_cremerlab(self, params):
        my_param_tbl = self._form_my_param_tbl(params, popt=self.tbl_popt)

        # !!DEBUGGING BY COMPARING THE CURVE FIT INPUT AND POPT VALUES

        param_cmp_ = self._form_param_tbl_comparison(my_param_tbl)

        # 2024-03-04 11:39:26 result differences
        # amplitude inputs are all the same, popt v different scale of 100 or so.
        cmp_amplitude = param_cmp_.filter(pl.col(self.__PARAM) == self.__AMPLITUDE)

        # all location inputs are within 1. popt out for all except peak 3 for some reason.
        cmp_location = param_cmp_.filter(pl.col(self.__PARAM) == self.__LOCATION)

        # my scale ub input for peak 3 is +50 mins on clab. biiiig difference.
        cmp_scale = param_cmp_.filter(pl.col(self.__PARAM) == self.__SCALE)

        cmp_skew = param_cmp_.filter(pl.col(self.__PARAM) == self.__SKEW)

    def pipe_fit_scores(self, x_unit):
        windowed_recon = (
            self.tbl_signals.pipe(pl.from_pandas)
            .filter(pl.col(self.__SIGNAL).is_in([self.__X, self.__RECON]))
            .pivot(
                index=[self.__W_TYPE, self.__W_IDX, x_unit],
                columns=self.__SIGNAL,
                values=self.__AMPLITUDE,
            )
            .to_pandas()
        )

        self.tbl_fit_report = calc_fit_scores(
            windowed_recon=windowed_recon,
            x_unit=x_unit,
        )

    def update_signal_tbl_with_recon(self, recon_df):
        self.tbl_signals = (
            self.tbl_signals.pipe(pl.from_pandas)
            .pivot(
                index=[self.__W_TYPE, self.__W_IDX, self.__UNIT_IDX, self.__TIME_UNIT],
                columns=self.__SIGNAL,
                values=self.__AMPLITUDE,
            )
            .hstack(recon_df.pipe(pl.from_pandas).select(self.__RECON))
            .melt(
                id_vars=[
                    self.__W_TYPE,
                    self.__W_IDX,
                    self.__UNIT_IDX,
                    self.__TIME_UNIT,
                ],
                value_vars=[self.__X, self.__X_CORRECTED, self.__RECON],
                variable_name=self.__SIGNAL,
                value_name=self.__AMPLITUDE,
            )
            .to_pandas()
        )

    def update_signal_tbl_with_windows(self):
        self.tbl_signals = (
            self.tbl_signals.pipe(pl.from_pandas)
            .join(
                self.X_w.pipe(pl.from_pandas).select(
                    [self.__W_TYPE, self.__W_IDX, self.__X_IDX]
                ),
                on=self.__X_IDX,
            )
            .select(
                [
                    self.__W_TYPE,
                    self.__W_IDX,
                    self.__SIGNAL,
                    self.__X_IDX,
                    self.__TIME_UNIT,
                    self.__AMPLITUDE,
                ]
            )
            .to_pandas()
            .rename({self.__X_IDX: self.__UNIT_IDX}, axis=1)
        )

    def prepare_tables(self, map_peaks_output):
        self.tbl_peak_idx_X_idx = (
            map_peaks_output[self.__MAXIMA]
            .pipe(pl.from_pandas)
            .filter(pl.col(self.__DIM).is_in([self.__X_IDX]))
            .pivot(index=self.__P_IDX, columns=self.__DIM, values=self.__VALUE)
            .select(pl.col(self.__P_IDX), pl.col(self.__X_IDX).cast(int).cast(str))
        )

        # mapping window type to index to X idx
        self.tbl_window_to_X_idx = self.X_w.pipe(pl.from_pandas).select(
            pl.col(self.__W_TYPE),
            pl.col(self.__W_IDX),
            pl.col(self.__X_IDX),
        )

        # mapping the windows to the peaks

        self.tbl_window_to_peak_idx = self.tbl_peak_idx_X_idx.join(
            self.tbl_window_to_X_idx.with_columns(pl.col(self.__X_IDX).cast(str)),
            on=self.__X_IDX,
        )

        # TODO: reorganise all data into tbls of 'p_idx','param','dim','unit','value', in that order. that precludes 'idx' as a unit, and rather an id, since the x ('mins') and y ('amp') is both accessed by an idx.

        maxima = self.prepare_maxima_tbl(map_peaks_output)

        widths = map_peaks_output["widths"].pipe(pl.from_pandas)

        contour_line_bounds = self.prepare_contour_line_bounds(map_peaks_output)

        # join all the tables back together, with a null 'side' column in widths and maxima to conform to the contour_line_bounds tbl shape.
        self.tbl_peak_map = self.prepare_tbl_peak_map(maxima=maxima, widths=widths, contour_line_bounds=contour_line_bounds)

    def prepare_contour_line_bounds(self, map_peaks_output):
        contour_line_bounds = (
            map_peaks_output["contour_line_bounds"]
            .pipe(pl.from_pandas)
            .rename({self.__LOC: self.__SIDE})
            .pivot(
                index=[self.__P_IDX, self.__SIDE, self.__MSNT],
                columns=self.__DIM,
                values=self.__VALUE,
            )
            .select(
                pl.col([self.__P_IDX, self.__SIDE, self.__MSNT]),
                pl.col(self.__X_IDX_ROUNDED).cast(int).alias(self.__IDX),
                pl.col(self.__TIME_UNIT).alias(self.input_keys_time),
                pl.col(self.__X),
            )
            .melt(
                id_vars=[self.__P_IDX, self.__IDX, self.__SIDE, self.__MSNT],
                value_vars=[self.input_keys_time, self.__X],
                variable_name=self.__UNIT,
                value_name=self.__VALUE,
            )
        )
        return contour_line_bounds

    def _form_param_tbl_comparison(self, my_param_tbl):
        _my_param_tbl_anti = my_param_tbl.with_row_index().join(
            self.cremerlab_results.tbls.param_tbl,
            on=["peak_idx_abs", "w_type", "w_idx", "p_idx", "param", "type"],
            how="anti",
        )
        # rows in cl_param_tbl whose key is not present in my_param_tbl
        _cl_param_tbl_anti = (
            self.cremerlab_results.tbls.param_tbl.with_row_index().join(
                my_param_tbl,
                on=["peak_idx_abs", "w_type", "w_idx", "p_idx", "param", "type"],
                how="anti",
            )
        )

        param_cmp = my_param_tbl.join(
            self.cremerlab_results.tbls.param_tbl,
            on=["peak_idx_abs", "w_type", "w_idx", "p_idx", "param", "type"],
            how="left",
        )

        param_cmp_ = param_cmp.with_columns(
            pl.col("mine").ne(pl.col("clab")).alias("is_diff"),
        ).with_columns(
            pl.struct(["mine", "clab"])
            .map_elements(
                lambda cols: np.isclose(cols["mine"], cols["clab"], atol=10e-1)
            )
            .cast(bool)
            .alias("is_close")
        )

        return param_cmp_

    def _form_my_param_tbl(self, params, popt):
        """ """

        my_curve_input = params.pipe(pl.from_pandas).with_columns(
            pl.col("p_idx").cast(str).alias("peak_idx_abs"),
            pl.col("param")
            .cast(str)
            .replace({"maxima": "amplitude", "loc": "location"}),
        )

        my_popt = (
            popt.with_columns(pl.col("p_idx").cast(str).alias("peak_idx_abs"))
            .rename({"maxima": "amplitude", "loc": "location"})
            .melt(
                id_vars="peak_idx_abs",
                value_vars=["amplitude", "location", "scale", "skew"],
                variable_name="param",
                value_name="popt",
            )
        )

        my_param_tbl = (
            my_curve_input.join(
                my_popt,
                on=["peak_idx_abs", "param"],
                how="left",
            )
            .with_columns(
                pl.lit("mine").alias("source"),
                pl.col("p_idx").rank("dense").over("w_idx").sub(1).cast(int),
            )
            .melt(
                id_vars=["peak_idx_abs", "w_type", "w_idx", "p_idx", "param"],
                value_vars=["lb", "p0", "ub", "popt"],
                variable_name="type",
                value_name="mine",
            )
        )

        return my_param_tbl

    def curve_fit_input_pipeline(
        self,
        peak_map,  #: DataFrame[dc_schs.PeakMsntsWindowed],
        X_windowed: DataFrame[mw_schs.X_Windowed],
        x_unit: str,
    ):
        """
        2024-03-02 00:52:05

        `params_factory` doesnt need the dim or unit columns, and only operates on the maxima x and y and whh_width. So first filter to that. index = ['w_idx','p_idx','msnt','type','unit']. select here what to submit to dataprepper, and modify dataprepper to be agnostic to unit type.
        """

        pf_input = (
            peak_map.filter(
                pl.col(self.__MSNT).is_in([self.__MAXIMA, self.__WHH]),
                pl.col(self.__TYPE).is_in([self.__POINT, self.__WIDTH]),
                pl.col(self.__UNIT).is_in([self.__AMPLITUDE, x_unit]),
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col(self.__MSNT),
                        pl.lit("_"),
                        pl.col(self.__DIM),
                    ]
                ).alias("pivot_cols")
            )
            .pivot(
                index=[self.__W_TYPE, self.__W_IDX, self.__P_IDX],
                columns=["pivot_cols"],
                values=self.__VALUE,
                aggregate_function="first",
            )
            .rename(
                {
                    self.__MAXIMA + "_" + self.__Y: self.__AMPLITUDE,
                    self.__MAXIMA + "_" + self.__TIME_UNIT: self.__LOC,
                    self.__WHH + "_" + self.__TIME_UNIT: self.__SCALE,
                }
            )
        )

        params: DataFrame[dc_schs.Params] = opt_params.params_factory(
            peak_msnts_windowed=pf_input,
            x_unit=x_unit,
            X_w=X_windowed,
            timestep=self._timestep,
        )

        return params

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
        x_unit: str,
    ):
        """
        From peak map we require the peak location, maxima, and WHH. Provide them as a table called 'OptParamPeakInput'. Provide it in long form.
        """

        # TODO: replace the class call with the function calls

        fit_func = deconvolution.FitFuncReg("scipy").fit_func
        opt_func = deconvolution.OptFuncReg("scipy").opt_func

        popt = deconvolution.popt_factory(
            X_w=X_windowed,
            params=params,
            optimizer=opt_func,
            fit_func=fit_func,
            max_nfev=1e6,
            x_key=x_unit,
        )

        return popt

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
            .with_row_index(name=self.keys_X_idx)
            .select(
                [
                    pl.col(self.keys_X_idx).cast(int),
                    pl.col(key_time).alias(self.input_keys_time),
                    pl.col(key_amp).alias(self.keys_X),
                ]
            )
        )

        correct_baseline_input = (
            signal_store.select(pl.col("X_idx"), pl.col("X"))
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
            .select(
                pl.col(self.keys_X).alias(self.keys_X_corrected),
                pl.col(self.keys_X_idx),
            )
        )

        signal_store_: pd.DataFrame = (
            signal_store.join(X_corrected, on=self.keys_X_idx)
            .select(
                [
                    pl.col(self.keys_X_idx),
                    pl.col(self.input_keys_time),
                    pl.col(self.keys_X),
                    pl.col(self.keys_X_corrected),
                ]
            )
            .melt(
                id_vars=[self.keys_X_idx, self.input_keys_time],
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
            .filter(pl.col(bcorr_defs.KEY_SIGNAL) == self.keys_active_signal)
            .pivot(index=self.keys_X_idx, columns="signal", values="amplitude")
            .select(
                pl.col(self.keys_X_idx),
                pl.col(self.keys_X_corrected).alias(self.keys_X),
            )
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
            .filter(pl.col(self.__SIGNAL) == self.keys_active_signal)
            .pivot(
                index=self.keys_X_idx, columns=self.__SIGNAL, values=self.__AMPLITUDE
            )
            .select(
                pl.col(self.keys_X_idx),
                pl.col(self.keys_X_corrected).alias(self.keys_X),
            )
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
            # starts with a long frame of 5 columns: p_idx, loc, msnt, dim, value
            peak_map.contour_line_bounds.pipe(pl.from_pandas)
            # pivot on dim with index p_idx, loc, msnt
            .pivot(
                index=[
                    self.__P_IDX,
                    self.__LOC,
                    self.__MSNT,
                ],
                columns=self.__DIM,
                values=self.__VALUE,
            )
            # add new column 'time' as transform of X_idx_rounded
            .with_columns(
                pl.col(self.__X_IDX_ROUNDED)
                .mul(self._timestep)
                .alias(self.input_keys_time),
            )
            # melt back to long form
            .melt(
                id_vars=[self.__P_IDX, self.__LOC, self.__MSNT],
                variable_name=self.__DIM,
                value_name=self.__VALUE,
            ).to_pandas()
        )

        maxima = (
            peak_map.maxima.pipe(pl.from_pandas)
            .pivot(
                index=[self.__P_IDX, self.__LOC],
                columns=self.__DIM,
                values=self.__VALUE,
            )
            .select(
                pl.col(self.__P_IDX),
                pl.col(self.__LOC).alias(self.__PARAM),
                pl.col(self.__X_IDX),
                pl.col(self.__X_IDX).mul(self._timestep).alias(self.input_keys_time),
                pl.col(self.__X),
            )
            .melt(
                id_vars=[self.__P_IDX, self.__PARAM],
                variable_name=self.__DIM,
                value_name=self.__VALUE,
            )
            .to_pandas()
        )

        widths = (
            peak_map.widths.pipe(pl.from_pandas)
            .select(
                pl.col(self.__P_IDX),
                pl.col(self.__MSNT),
                pl.col(self.__VALUE).alias(self.__X_IDX),
                pl.col(self.__VALUE).mul(self._timestep).alias(self.input_keys_time),
            )
            .melt(
                id_vars=[self.__P_IDX, self.__MSNT],
                variable_name=self.__UNIT,
                value_name=self.__VALUE,
            )
            .to_pandas()
        )

        peak_map = {
            self.__CONTOUR_LINE_BOUNDS: contour_line_bounds,
            self.__MAXIMA: maxima,
            self.__WIDTHS: widths,
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

    def prepare_maxima_tbl(self, map_peaks_output):
        maxima: pl.DataFrame = (
            map_peaks_output[self.__MAXIMA].pipe(pl.from_pandas)
            .pivot(index=[self.__P_IDX, self.__PARAM], columns=self.__DIM, values=self.__VALUE)
            .rename({
                self.__X_IDX:self.__IDX,
            })
            .melt(id_vars=[self.__P_IDX,self.__PARAM], value_vars=[self.__IDX,self.input_keys_time,self.__X], variable_name=self.__UNIT, value_name=self.__VALUE)
            .with_columns(
                pl.when(pl.col(self.__UNIT).is_in([self.__IDX,self.input_keys_time])).then(pl.lit(self.input_keys_time)).when(pl.col(self.__UNIT)==self.__X).then(pl.lit(self.__Y)).alias(self.__DIM)
            )
            .pivot(index=[self.__P_IDX,self.__PARAM], columns=self.__UNIT, values=self.__VALUE)
            .with_columns(pl.col(self.__IDX).cast(int))
            .melt(
                id_vars=[self.__P_IDX,self.__IDX,self.__PARAM], value_vars=[self.input_keys_time,self.__X],variable_name=self.__UNIT,value_name=self.__VALUE
            )
            .rename({self.__PARAM:self.__MSNT})
        )  # fmt: skip

        return maxima

    def prepare_tbl_peak_map(self,
                             maxima,
                             widths,
                             contour_line_bounds,
                             ):
        
        tbl_peak_map = (
            # join maxima, widths and contour line tables
            pl.concat(
                [
                    maxima.with_columns(
                        pl.col([self.__P_IDX]),
                        pl.lit(None).alias(self.__SIDE),
                        pl.col([self.__MSNT, self.__UNIT, self.__VALUE]),
                    ),
                    widths.with_columns(
                        pl.col([self.__P_IDX]),
                        pl.lit(None).alias(self.__SIDE),
                        pl.col([self.__MSNT, self.__UNIT, self.__VALUE]),
                    ),
                    contour_line_bounds,
                ],
                how="diagonal_relaxed",
            )
            # select columns, excluding 'idx'.
            .select(self.__P_IDX, self.__MSNT, self.__SIDE, self.__UNIT, self.__VALUE)
            # add window type, idx to peak map
            .pipe(
                lambda df: self.tbl_window_to_peak_idx.select(
                    self.__W_TYPE, self.__W_IDX, self.__P_IDX
                ).join(df, on=self.__P_IDX, how="left")
            )
            # add dim column containing the dimension of the measurement, x or y.
            .with_columns(
                pl.when(pl.col(self.__UNIT).is_in([self.input_keys_time, self.__X_IDX]))
                .then(pl.lit(self.__TIME_UNIT))
                .when(pl.col(self.__UNIT) == self.__X)
                .then(pl.lit(self.__Y))
                .otherwise(pl.lit(self.__UNASSIGNED))
                .alias(self.__DIM)
            )
            # want to add a unitcolumn for all x measurements currently in time units. do this by pivoting
            # on unit then adding a new column 'unit_idx' via transformation
            # pivot on 'unit' with index of 'w_type','w_idx','p_idx','msnt',side','dim' and values from 'value'
            .pivot(
                index=[
                    self.__W_TYPE,
                    self.__W_IDX,
                    self.__P_IDX,
                    self.__MSNT,
                    self.__SIDE,
                    self.__DIM,
                ],
                columns=self.__UNIT,
                values=self.__VALUE,
            )
            # replace the X field with "amp", a synonym
            .rename({self.__X: self.__AMPLITUDE})
            # add unit_idx column based on transformation from time unit, division by timestep
            .with_columns(
                pl.col(self.input_keys_time)
                .truediv(self._timestep)
                .round(0)
                .cast(int)
                .alias(self.__UNIT_IDX)
            )
            # return to long form, excluding X_idx column which contains incorrect values.
            .melt(
                id_vars=[
                    self.__W_TYPE,
                    self.__W_IDX,
                    self.__P_IDX,
                    self.__MSNT,
                    self.__SIDE,
                    self.__DIM,
                ],
                value_vars=[self.input_keys_time, self.__AMPLITUDE, self.__UNIT_IDX],
                value_name=self.__VALUE,
                variable_name=self.__UNIT,
            )
            # the transformations create spurious null cells which need tobe removed
            .filter(~pl.col(self.__VALUE).is_null())
            # add new columns
            .with_columns(
                # create a new column: type_ describing the type of the measurement - width, point, contour_line_bound
                pl.when(pl.col(self.__MSNT).str.contains(self.__WIDTH))
                .then(pl.lit(self.__WIDTH))
                .when(pl.col(self.__MSNT) == self.__MAXIMA)
                .then(pl.lit(self.__POINT))
                .when(pl.col(self.__SIDE).is_not_null())
                .then(pl.lit(self.__CONTOUR_LINE_BOUND))
                .otherwise(pl.lit(self.__UNASSIGNED))
                .alias(self.__TYPE),
                # within msnt column, remove prefix "width_" as its information is now contained in type_
                pl.col(self.__MSNT).str.replace("width_", ""),
            )
        )
        return tbl_peak_map


class Tbls(SimpleNamespace):
    pass
