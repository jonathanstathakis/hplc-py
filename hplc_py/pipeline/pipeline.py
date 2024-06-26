from hplc_py.deconvolution import analysis
import warnings
from tqdm import tqdm
from types import SimpleNamespace
from typing import Any, Callable, Type
import hvplot.pandas
import numpy as np
import pandas as pd
import polars as pl
from cachier import cachier
from numpy.typing import ArrayLike
from pandera.typing import DataFrame, Series

import hplc_py.map_peaks.definitions as mp_defs
from hplc_py.baseline_correction import definitions as bcorr_defs
from hplc_py.baseline_correction.baseline_correction import BaselineCorrection
from hplc_py.common import caching
from hplc_py.common import common_schemas as com_schs
from hplc_py.common import definitions as com_defs
from hplc_py.common.caching import CACHE_PATH, custom_param_hasher
from hplc_py.deconvolution import deconvolution, opt_params
from hplc_py.deconvolution import definitions as KeysFitReport
from hplc_py.deconvolution import schemas as dc_schs, definitions as dc_defs
from hplc_py.deconvolution.fit_assessment import Reporter
from hplc_py.map_peaks.map_peaks import PeakMapper
from hplc_py.map_peaks.viz import PeakMapPlotHandler
from hplc_py.map_windows import schemas as mw_schs
from hplc_py.map_windows.map_windows import MapWindows
from hplc_py.pipeline.hplc_result_analyzer import HPLCResultsAnalzyer
from hplc_py.pipeline import preprocessing
from dataclasses import dataclass


@dataclass(init=False)
class PipelineKeys:
    W_TYPE: str = "w_type"
    W_IDX: str = "w_idx"
    IDX: str = "idx"
    AMPLITUDE: str = "amplitude"
    UNIT_IDX: str = "unit_idx"
    keys_x_unit: str = ""
    SIGNAL: str = "signal"
    MAXIMA: str = "maxima"
    DIM: str = "dim"
    P_IDX: str = "p_idx"
    VALUE: str = "value"
    PARAM: str = "param"
    IDX: str = "idx"
    UNIT: str = "unit"
    Y: str = "y"
    X: str = "X"
    LOC: str = "loc"
    SIDE: str = "side"
    MSNT: str = "msnt"
    IDX_ROUNDED: str = "idx_rounded"
    UNASSIGNED: str = "unassigned"
    WIDTH: str = "width"
    POINT: str = "point"
    CONTOUR_LINE_BOUND: str = "contour_line_bound"
    TYPE: str = "type"
    RECON: str = "recon"
    X_CORRECTED: str = "X_corrected"
    LOCATION: str = "location"
    SCALE: str = "scale"
    SKEW: str = "skew"
    MIXED: str = "mixed"
    UNMIXED: str = "unmixed"
    WHH: str = "whh"
    CONTOUR_LINE_BOUNDS: str = "contour_line_bounds"
    WIDTHS: str = "widths"


class DeconvolutionPipeline:
    """
    The deconvolution pipeline class, executing the pipeline and storing the state information of the pipeline.

    Will likely integrate the dashboard into this class.

    TODO:
    - add dashboard functionality
    """

    def __init__(
        self,
    ):

        # internally defined access keys

        self.keys_X = com_defs.X
        self.keys_idx = com_defs.IDX
        self.keys_time = "time"
        self.keys_X_corrected = com_defs.X + "_corrected"

        self.keys = PipelineKeys()

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
            n_iter=55,
            verbose=True,
        ),
        debug: bool = False,
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
        self.tbl_input_data = data

        # which_x_unit = "unit_idx"
        self.keys.keys_x_unit = self.set_x_unit_key()
        # calculate timestep
        self._timestep = calculate_timestep(data[self.input_keys_time])
        # if debug:
        #     self.get_hplc_results()

        ##### Ingestion #####
        signal_store = self.ingest_data(data)

        ##### Preprocessing #####
        preprocessing_input = (
            signal_store.pipe(pl.from_pandas)
            .select(pl.col("idx"), pl.col("X"))
            .to_pandas()
            .pipe(DataFrame[com_schs.X_Schema])
        )

        preprocessor = preprocessing.PreprocessData(
            correct_baseline=correct_baseline, bcorr_kwargs=bcorr_kwargs
        )

        preprocessor.fit(X=preprocessing_input)
        X_processed = preprocessor.transform()

        signal_store: pd.DataFrame = (
            signal_store.pipe(pl.from_pandas)
            .join(X_processed.pipe(pl.from_pandas), on=self.keys_idx)
            .select(
                [
                    pl.col(self.keys_idx),
                    pl.col(self.keys_time),
                    pl.col(self.keys_X),
                    pl.col(self.keys_X_corrected),
                ]
            )
            .melt(
                id_vars=[self.keys_idx, self.keys_time],
                variable_name="signal",
                value_name="amplitude",
            )
            .to_pandas()
        )

        if correct_baseline:
            self.keys_active_signal = self.keys_X_corrected
        else:
            self.keys_active_signal = self.keys_X

        ##### Map Peaks #####

        map_peaks_output: dict
        peak_map_plot_handler: Any
        map_peaks_output, peak_map_plot_handler = self.pipe_map_peaks(
            signal_store=signal_store,
            find_peaks_kwargs=find_peaks_kwargs,
        )

        ##### Map Windows #####
        self.X_w, peak_window_spans = self.pipe_window_X(
            store_signals=signal_store,
            contour_line_bounds=map_peaks_output["contour_line_bounds"],
        )
        ##########################################################
        # prepare normalised peak : window tbl and window time / index bounds table

        signal_store = signal_store.pipe(
            self.update_signal_tbl_with_windows, X_w=self.X_w
        )

        # normalized or otherwise core tables primarily for join operations (?)

        self.prepare_tables(map_peaks_output)

        ##### Deconvolution #####

        # creating the "X_windowed" input from the signal store based on the selected `x_unit`

        X_windowed_input = self.tbl_signal_mixed.pipe(
            self.get_X_windowed,
            x_unit=self.keys.keys_x_unit,
            active_signal=self.keys_active_signal,
        )

        # calculate timestep
        self._timestep = calculate_timestep(data[self.input_keys_time])

        params: DataFrame[dc_schs.Params] = self.curve_fit_input_pipeline(
            peak_map=self.tbl_peak_map,
            X_windowed=X_windowed_input,
            x_unit=self.keys.keys_x_unit,
        )

        popt: DataFrame[dc_schs.Popt] = self.pipe_deconvolution(
            X_windowed=X_windowed_input, params=params, x_unit=self.keys.keys_x_unit
        )

        self.tbl_popt = popt

        reconstructor_in_signal = X_windowed_input.pipe(
            dc_schs.ReconstructorSignalIn.validate
        ).pipe(DataFrame[dc_schs.ReconstructorSignalIn])

        reconstructor_in_popt = popt.pipe(dc_schs.ReconstructorPoptIn.validate).pipe(
            DataFrame[dc_schs.ReconstructorPoptIn]
        )

        # popt calculated earlier

        reconstructor = analysis.Reconstructor(
            X_w=reconstructor_in_signal,
            popt=reconstructor_in_popt,
            x_unit=self.keys.keys_x_unit,
        )

        self.tbl_signal_unmixed = reconstructor.unmixed_signals
        recon_df = reconstructor.mixed_signals

        # add recon to signal store

        self.tbl_signal_mixed = self.tbl_signal_mixed.pipe(
            self.update_signal_tbl_with_recon, recon_df
        )

        # get_fit_report wants tbl_signals as the input. tbl_signals is a 6 column pandas dataframe of w_type, w_idx, unit_idx, signal, amplitude. signal contains X, X_corrected, recon. What it actually does is filter tbl signals for X and recon.

        # calc_fit_report_in = self.get_calc_fit_report_in(
        #     tbl_signals=self.tbl_signal_mixed
        # )

        analyzer_in = self.tbl_signal_mixed.pipe(
            deconvolution.get_active_signal_as_mixed,
            x_unit=self.keys.keys_x_unit,
            active_signal=self.keys_active_signal,
            keys=dc_defs.keys_tbl_mixed_signal,
        )

        analyzer = analysis.Analyzer(
            data=analyzer_in,
            x_unit=self.keys.keys_x_unit,
        )

        self.tbl_fit_report = analyzer.get_fit_report()

        END_OF_PIPELINE = ""

        breakpoint()
        # !! DEBUGGING

        # !! input hplc params into my deconvolution pipeline and see how the results differ
        if debug:
            self.pipe_results_comparison_with_cremerlab(params)

        return None

    def set_x_unit_key(self):
        which_x_unit = self.input_keys_time
        if which_x_unit == "unit_idx":
            keys_x_unit = "unit_idx"
        elif which_x_unit == self.input_keys_time:
            keys_x_unit = self.input_keys_time
        else:
            raise ValueError("please input one of 'unit_idx', self.input_keys_time")

        return keys_x_unit

    def get_X_windowed(
        self,
        tbl_signal_mixed,
        x_unit: str,
        active_signal: str,
    ):

        X_windowed_input = (
            tbl_signal_mixed.pipe(pl.from_pandas)
            .filter(pl.col(self.keys.SIGNAL).eq(active_signal))
            .select([self.keys.W_TYPE, self.keys.W_IDX, x_unit, self.keys.AMPLITUDE])
            .to_pandas()
        )

        return X_windowed_input

    def viz_popt_statistics(self, results):
        """
        Observe the distribution of parameter values as functions of p_idx and param as a box plot and a table
        of mean, std, rstd.
        """

        index = ["w_type", "w_idx", "p_idx", "param"]
        popt_results = results.sort(index).filter(pl.col("nfev_index") != "-1")

        popt_boxplot = popt_results.plot.box(y="p0", by="p_idx", groupby="param")
        hvplot.show(popt_boxplot)

        popt_stats = popt_results.groupby(index).agg(
            pl.col("p0").mean().alias("mean"),
            pl.col("p0").std().alias("std"),
            pl.col("p0").std().truediv(pl.col("p0").mean()).alias("rstd"),
        )

    def pipe_results_comparison_with_cremerlab(self, params):
        my_param_tbl = self._form_my_param_tbl(params, popt=self.tbl_popt)

        # !!DEBUGGING BY COMPARING THE CURVE FIT INPUT AND POPT VALUES

        param_cmp_ = self._form_param_tbl_comparison(my_param_tbl)

        # 2024-03-04 11:39:26 result differences
        # amplitude inputs are all the same, popt v different scale of 100 or so.
        cmp_amplitude = param_cmp_.filter(
            pl.col(self.keys.PARAM) == self.keys.AMPLITUDE
        )

        # all location inputs are within 1. popt out for all except peak 3 for some reason.
        cmp_location = param_cmp_.filter(pl.col(self.keys.PARAM) == self.keys.LOCATION)

        # my scale ub input for peak 3 is +50 mins on clab. biiiig difference.
        cmp_scale = param_cmp_.filter(pl.col(self.keys.PARAM) == self.keys.SCALE)

        cmp_skew = param_cmp_.filter(pl.col(self.keys.PARAM) == self.keys.SKEW)

    def get_fit_report(self, tbl_signals):

        mixed_and_unmixed = tbl_signals.pipe(self.get_calc_fit_report_in).pipe(
            calc_fit_report
        )

        return fit_report.pipe(pl.from_pandas)

    def get_calc_fit_report_in(
        self, data: pd.DataFrame
    ) -> DataFrame[dc_schs.ActiveSignal]:

        out = data.pipe(lambda df: df if breakpoint() else df)

    def update_signal_tbl_with_recon(
        self,
        tbl_signals,
        recon_df,
    ):

        recon_idx = [self.keys.W_TYPE, self.keys.W_IDX, self.keys.keys_x_unit]
        tbl_signal_idx = [
            self.keys.W_TYPE,
            self.keys.W_IDX,
            self.keys.UNIT_IDX,
            self.keys.keys_x_unit,
        ]
        post_join_value_cols = [self.keys.X, self.keys.X_CORRECTED, self.keys.RECON]

        updated_tbl = (
            tbl_signals.pipe(pl.from_pandas)
            .pivot(
                index=tbl_signal_idx,
                columns=self.keys.SIGNAL,
                values=self.keys.AMPLITUDE,
            )
            # select the amplitude column as "recon" before joining
            .join(
                recon_df.filter(pl.col(self.keys.SIGNAL) == self.keys.RECON).select(
                    pl.col(recon_idx),
                    pl.col(self.keys.AMPLITUDE).alias(self.keys.RECON),
                ),
                on=recon_idx,
            )
            .melt(
                id_vars=tbl_signal_idx,
                value_vars=post_join_value_cols,
                variable_name=self.keys.SIGNAL,
                value_name=self.keys.AMPLITUDE,
            )
        )

        return updated_tbl

    def update_signal_tbl_with_windows(self, tbl_signal_mixed, X_w):
        tbl_signal_mixed = (
            tbl_signal_mixed.pipe(pl.from_pandas)
            .join(
                X_w.pipe(pl.from_pandas).select(
                    [self.keys.W_TYPE, self.keys.W_IDX, self.keys.IDX]
                ),
                on=self.keys.IDX,
            )
            .select(
                [
                    self.keys.W_TYPE,
                    self.keys.W_IDX,
                    self.keys.SIGNAL,
                    self.keys.IDX,
                    self.keys.keys_x_unit,
                    self.keys.AMPLITUDE,
                ]
            )
            .to_pandas()
            .rename({self.keys.IDX: self.keys.UNIT_IDX}, axis=1)
        )
        return tbl_signal_mixed

    def prepare_tables(self, map_peaks_output):

        self.tbl_peak_idx_X_idx = (
            map_peaks_output[self.keys.MAXIMA]
            .pipe(pl.from_pandas)
            .filter(pl.col(self.keys.DIM).eq(self.keys.IDX))
            .select(
                pl.col(self.keys.P_IDX),
                pl.col(self.keys.VALUE).cast(int).alias(self.keys_idx),
            )
        )

        # mapping window type to index to X idx
        self.tbl_window_to_X_idx = self.X_w.pipe(pl.from_pandas).select(
            pl.col(self.keys.W_TYPE),
            pl.col(self.keys.W_IDX),
            pl.col(self.keys.IDX),
        )

        # mapping the windows to the peaks

        self.tbl_window_to_peak_idx = self.tbl_peak_idx_X_idx.join(
            self.tbl_window_to_X_idx.with_columns(pl.col(self.keys.IDX)),
            on=self.keys.IDX,
        )

        maxima = self.prepare_maxima_tbl(map_peaks_output)

        widths = map_peaks_output["widths"].pipe(pl.from_pandas)

        contour_line_bounds = self.prepare_contour_line_bounds(map_peaks_output)

        # join all the tables back together, with a null 'side' column in widths and maxima to conform to the contour_line_bounds tbl shape.
        self.tbl_peak_map = self.prepare_tbl_peak_map(
            maxima=maxima, widths=widths, contour_line_bounds=contour_line_bounds
        )

    def prepare_contour_line_bounds(self, map_peaks_output):
        contour_line_bounds = (
            map_peaks_output["contour_line_bounds"]
            .pipe(pl.from_pandas)
            .rename({self.keys.LOC: self.keys.SIDE})
            .pivot(
                index=[self.keys.P_IDX, self.keys.SIDE, self.keys.MSNT],
                columns=self.keys.DIM,
                values=self.keys.VALUE,
            )
            .select(
                pl.col([self.keys.P_IDX, self.keys.SIDE, self.keys.MSNT]),
                pl.col(self.keys.IDX_ROUNDED).cast(int).alias(self.keys.IDX),
                pl.col(self.keys.keys_x_unit).alias(self.input_keys_time),
                pl.col(self.keys.X),
            )
            .melt(
                id_vars=[
                    self.keys.P_IDX,
                    self.keys.IDX,
                    self.keys.SIDE,
                    self.keys.MSNT,
                ],
                value_vars=[self.input_keys_time, self.keys.X],
                variable_name=self.keys.UNIT,
                value_name=self.keys.VALUE,
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

        not_close = param_cmp_.filter(pl.col("is_close").not_())

        return param_cmp_

    def _form_my_param_tbl(self, params, popt: pl.DataFrame):
        """ """

        my_curve_input = params.pipe(pl.from_pandas).with_columns(
            pl.col("p_idx").cast(str).alias("peak_idx_abs"),
            pl.col("param")
            .cast(str)
            .replace({"maxima": "amplitude", "loc": "location"}),
        )

        my_popt = (
            popt.pipe(
                lambda df: (
                    df.pipe(pl.from_pandas) if isinstance(df, pd.DataFrame) else df
                )
            )
            .with_columns(pl.col("p_idx").cast(str).alias("peak_idx_abs"))
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
                pl.col(self.keys.MSNT).is_in([self.keys.MAXIMA, self.keys.WHH]),
                pl.col(self.keys.TYPE).is_in([self.keys.POINT, self.keys.WIDTH]),
                pl.col(self.keys.UNIT).is_in([self.keys.AMPLITUDE, x_unit]),
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col(self.keys.MSNT),
                        pl.lit("_"),
                        pl.col(self.keys.DIM),
                    ]
                ).alias("pivot_cols")
            )
            .pivot(
                index=[self.keys.W_TYPE, self.keys.W_IDX, self.keys.P_IDX],
                columns=["pivot_cols"],
                values=self.keys.VALUE,
                aggregate_function="first",
            )
            .rename(
                {
                    self.keys.MAXIMA + "_" + self.keys.Y: self.keys.AMPLITUDE,
                    self.keys.MAXIMA + "_" + self.keys.keys_x_unit: self.keys.LOC,
                    self.keys.WHH + "_" + self.keys.keys_x_unit: self.keys.SCALE,
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
            data=self.tbl_input_data, cols={"time": "x", "signal": "y"}
        )

    def pipe_deconvolution(
        self,
        params,
        X_windowed: DataFrame[mw_schs.X_Windowed],
        x_unit: str,
        display_results: bool = False,
        max_nfev: int = int(1e6),
        n_interms: int = 5,
    ):
        """
            From peak map we require the peak location, maxima, and WHH. Provide them as a table called 'OptParamPeakInput'. Provide it in long form.

        stackoverflow.com/questions/54560909/advanced-curve-fitting-methods-that-allows-real-time-monitoring)
        TODO: define column accessors as ENUM passed through functions.
        """

        fit_func = deconvolution.FitFuncReg("scipy").fit_func
        opt_func = deconvolution.OptFuncReg("scipy").opt_func

        # cast param cat datatype to string to make join statements simpler

        param = "param"
        w_type = "w_type"
        w_idx = "w_idx"
        p_idx = "p_idx"
        lb = "lb"
        ub = "ub"
        p0_key = "p0"
        arg_input = "arg_input"
        value = "value"
        nfev_index_key = "nfev_index"
        w_type = "w_type"
        w_idx = "w_idx"
        p_idx = "p_idx"
        param = "param"
        value = "value"
        nfev_key = "nfev"
        mesg = "mesg"
        ier = "ier"
        col = "col"
        fvec = "fvec"
        success = "success"

        idx_cols = [w_type, w_idx, p_idx, param]
        params_val_cols = [lb, ub, p0_key]
        params_var_name = arg_input
        param_value_name = value
        params = (
            params.pipe(pl.from_pandas)
            .with_columns(
                pl.col("param").cast(str),
                pl.col("p_idx").cast(str),
                pl.col("w_idx").cast(str),
            )
            .melt(
                id_vars=idx_cols,
                value_vars=params_val_cols,
                variable_name=params_var_name,
                value_name=param_value_name,
            )
        )

        X_w: pl.DataFrame = X_windowed.pipe(pl.from_pandas)

        idx_keys = [nfev_index_key, w_type, w_idx, p_idx]
        int_idx_keys = [nfev_index_key, w_idx, p_idx]

        idx_schema = dict(zip(idx_keys, [str] * len(idx_keys)))

        schemas = dict(
            results_df={
                **idx_schema,
                param: str,
                "p0": float,
            },
            info_df={
                **idx_schema,
                nfev_key: int,
                mesg: str,
                ier: int,
                success: bool,
            },
            pcov_df={
                **idx_schema,
                col: str,
                value: float,
            },
            fvec_df={
                **idx_schema,
                fvec: float,
            },
        )
        # trying to identify a 'minimum' run time.
        """
        # Curve Fitting Through Least Squares
        
        [least_squares docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares)
        
        For tolerance hyperparameters, if the change is less than the value provided, termination occurs.
        
        Curve fit through least squares minimization has several termination conditions:
            - ftol: 
                - change of cost function - defaults to 1E-8, stopping when dF < ftol * F and when 'local quadratic model' and true model are in agreement.
                - If None is passed, it is not used, UNLESS 'lm' method is used(?).
                - if 'lm', ftol must be higher than 'machine epsilon'
            - xtol:
                - change in independent variables, defaults to 1E-8. For 'lm', defined as $Delta < xtol * norm(xs)$ where Delta is a trust-region radius and xs is the value of x scaled according to `x_scale` hyperparameter.
                - same as ftol, if None is passed, it is not used, UNLESS 'lm' method is used(?).
                - also same as ftol, if 'lm', xtol needs to be bigger than machine epsilon.
            - gtol:
                - norm of the gradient. Defaults to 1E-8.
                - for 'lm', maximum absolute value of the cosine of angles between columns of the Jacobian and residual vector is less than zero, or residual vector is zero.
                - same as ftol and xtol, must be higher than machine epsilon if method is 'lm'.
        
        machine epsilon: the numerical precision of the computer due to floating point errors. Numpy contains `finfo` which provides the machine epsilon for a given datatype. On my computer at 2024/03/06 the eps is measures as 2.22E-16. Therefore the default 1E-8 is perfectly acceptable.
        
        And termination is reported through a status number:
        
        > -1 : improper input parameters status returned from MINPACK.
        > 0 : the maximum number of function evaluations is exceeded.
        > 1 : gtol termination condition is satisfied.
        > 2 : ftol termination condition is satisfied.
        > 3 : xtol termination condition is satisfied.
        > 4 : Both ftol and xtol termination conditions are satisfied.
        
        ## Our Implementation
        
        As we are using bounds, we are using the 'trf' methnod.
        
        ## Methods
        
        'lm': Levenberg-Marquardt algorithm, implemented by MINPACK.
        'trf': Trust Region Reflective.
        
        
        """
        # to measure the execution of 1 iteration of curve fitting we'll need to catch the error and measure the difference in time. This will not be accurate as the various calls and the try except operation will add time but it will give us a ballpark figure.

        elapsed_time_stats = self.estimate_curve_fit_time(
            params=params,
            X_w=X_w,
            x_unit=x_unit,
            fit_func=fit_func,
            opt_func=opt_func,
            schemas=schemas,
        )

        # warn the user how long one iteration will take.
        # TODO: include calc max_nfev * statistics to give user idea of maximum run time
        warnings.warn(f"one iteration: {elapsed_time_stats}")

        output: dict[str, pl.DataFrame] = self.curve_fit_windows(
            params=params,
            X_w=X_w,
            x_unit=x_unit,
            fit_func=fit_func,
            opt_func=opt_func,
            max_nfev=max_nfev,
            n_interms=n_interms,
            schemas=schemas,
            verbose=True,
            terminate_on_fit=True,
        )

        output = {
            k: df.with_columns(pl.col(int_idx_keys).cast(int))
            for k, df in output.items()
        }

        output = {k: df.sort(idx_keys, descending=False) for k, df in output.items()}

        if display_results:
            self.viz_popt_statistics(output)

            # provide the last p0 obtained, potentially not the best result, but a temporarily acceptable value
        # until we determine a method of identifying the popt, given that the variance between iterations
        # is tiny. Refer to `viz_popt_statistics`.

        popt = (
            output["results_df"]
            .pipe(deconvolution.select_popt)
            .to_pandas()
            .pipe(deconvolution.get_wide_popt)
        )

        # aly = analysis.Analyzer(X_w=X_windowed, popt=popt, x_unit="x")

        return popt

    # @cachier(hash_func=custom_param_hasher, cache_dir=CACHE_PATH)
    def curve_fit_windows(
        self,
        params: pl.DataFrame,
        X_w: pl.DataFrame,
        x_unit: str,
        fit_func: Callable,
        opt_func: Callable,
        max_nfev: int,
        n_interms: int,
        schemas: dict[str, Type],
        verbose=True,
        terminate_on_fit: bool = True,
    ):
        """
        Calculate the popt for each window using `scipy.optimize.least_squares`.

        iterates over each window in `w_idx` `X_w` and `params`, calls `least_squares` and returns the full
        output as a dict of dataframes. Refer to `least_squares`[docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)


        # Usage Notes

        For a given `max_nfev`, (Number of Function EValuations), can optionally break the process up into
        intermediate minimizations in order to provide insight into minimization progress for complex fits.
        This is done by specifying `n_interms` which will divide `max_nfev` into equal portions, collecting
        the results of each intermediate result for later observation. `n_interms` default = 1, no intermediates.

        TODO: remove external opt_func and fit_func regs, just add as args in functions.
        TODO: add jaxfit least squares
        TODO: add personalised fit report
            - how long it took to fit
            ...
        TODO: add fit assessment for intermediates. add plotting as well.
        TODO: test odd max_nfev, n_interms inputs
        TODO: add success to infodict

        If intermediates are specified, you can stop iteration once a successful fit is achieved, or continue
        until `max_nfev` is reached. This is done via the terminate_on_fit arg.

        Call heirarchy is as follows:

            1. curve_fit_windows
            2. popt_factory
            3. get_popt
                - iterates over the max_nfev divisions.
            4. curve_fit_
            5. least_squares


        """

        wdw_grpby = params.partition_by(
            [KeysFitReport.W_IDX], maintain_order=True, as_dict=True
        ).items()

        idx_keys = ["nfev_index", "w_type", "w_idx", "p_idx"]
        wdw_idx_keys = ["w_type", "w_idx", "p_idx"]

        # arrange columns so idx keys are first and everything else follows
        rearrange_col_expr = pl.col(idx_keys), pl.exclude(idx_keys)

        output = {k: pl.DataFrame(schema=schema) for k, schema in schemas.items()}

        wix: int
        wdw: pl.DataFrame

        if verbose:
            window_it = tqdm(wdw_grpby)
        else:
            window_it = wdw_grpby

        for wix, wdw in window_it:
            X_df = X_w.filter(pl.col("w_type") == "peak", pl.col("w_idx") == wix)

            curve_fit_output: dict[str, pl.DataFrame] = deconvolution.popt_factory(
                X=X_w,
                params=wdw,
                optimizer=opt_func,
                fit_func=fit_func,
                max_nfev=max_nfev,
                n_interms=n_interms,
                x_key=x_unit,
                verbose=verbose,
                terminate_on_fit=terminate_on_fit,
            )

            wdw_idx = wdw.select(pl.col(wdw_idx_keys)).unique()

            output_ = {
                k: df.join(wdw_idx, how="left", on="p_idx")
                for k, df in curve_fit_output.items()
            }

            output_ = {k: df.select(rearrange_col_expr) for k, df in output_.items()}

            output = {k: pl.concat([df, output[k]]) for k, df in output_.items()}

        return output

    def ingest_data(self, data) -> pd.DataFrame:
        signal_store: pd.DataFrame = (
            data.pipe(pl.from_pandas)
            .with_row_index(name=self.keys_idx)
            .select(
                [
                    pl.col(self.keys_idx).cast(int),
                    pl.col(self.input_keys_time).alias(self.keys_time),
                    pl.col(self.input_keys_amp).alias(self.keys_X),
                ]
            )
            .to_pandas()
        )

        return signal_store

    def pipe_window_X(
        self,
        store_signals,
        contour_line_bounds: pd.DataFrame,
    ) -> tuple[DataFrame[mw_schs.X_Windowed], Any]:
        """
        Window X based on peak interval overlaps
        """

        left_bases = self.get_base_side(
            df=contour_line_bounds,
            side=mp_defs.KEY_LEFT,
            msnt=mp_defs.KEY_PB,
            unit=mp_defs.KEY_IDX_ROUNDED,
        )

        right_bases = self.get_base_side(
            df=contour_line_bounds,
            side=mp_defs.KEY_RIGHT,
            msnt=mp_defs.KEY_PB,
            unit=mp_defs.KEY_IDX_ROUNDED,
        )

        X_in: DataFrame[com_schs.X_Schema] = self.get_X(
            active_signal=self.keys_active_signal
        )

        window_mapper = MapWindows(left_bases=left_bases, right_bases=right_bases)

        X_windowed = window_mapper.fit_transform(X=X_in).pipe(pl.from_pandas)
        # write a new spans generator

        # See: https://holoviews.org/reference/elements/matplotlib/VSpans.html
        # spans uses arrays of x_start and x_end
        # spans uses

        window_spans = window_mapper.viz()
        hvplot.show(window_spans)
        return X_windowed, window_spans

    def get_X(self, active_signal):
        X: DataFrame[com_schs.X_Schema] = (
            self.signal.pipe(pl.from_pandas)
            .filter(pl.col(bcorr_defs.KEY_SIGNAL) == active_signal)
            .pivot(index=self.keys_idx, columns="signal", values="amplitude")
            .select(
                pl.col(self.keys_idx),
                pl.col(self.keys_X_corrected).alias(self.keys_X),
            )
            .to_pandas()
            .pipe(DataFrame[com_schs.X_Schema])
        )

        return X

    def get_base_side(self, df, side, msnt, unit):
        base = (
            df.pipe(pl.from_pandas)
            .filter(
                pl.col(mp_defs.KEY_MSNT) == msnt,
                pl.col(mp_defs.DIM) == unit,
                pl.col(mp_defs.LOC) == side,
            )
            .select(mp_defs.VALUE)
            .to_series()
            .to_pandas()
            .rename(f"{msnt}_{side}")
            .pipe(Series[float])
        )
        return base

    def pipe_map_peaks(
        self,
        signal_store,
        find_peaks_kwargs: mp_defs.MapPeaksKwargs = mp_defs.map_peaks_kwargs_defaults,
        viz_maxima: bool = True,
        viz_whh: bool = False,
        viz_bases: str = "pb",
    ) -> tuple[dict, PeakMapPlotHandler]:

        X: DataFrame[com_schs.X_Schema] = (
            signal_store.pipe(pl.from_pandas)
            .filter(pl.col(self.keys.SIGNAL) == self.keys_active_signal)
            .pivot(
                index="idx",
                columns=self.keys.SIGNAL,
                values=self.keys.AMPLITUDE,
            )
            .select(
                pl.col("idx"),
                pl.col(self.keys_X_corrected).alias(self.keys_X),
            )
            .to_pandas()
            .pipe(DataFrame[com_schs.X_Schema])
        )

        peak_mapper = PeakMapper(find_peaks_kwargs=find_peaks_kwargs)
        peak_mapper.fit_transform(X=X)
        peak_map = peak_mapper.peak_map_

        peak_map_plots = peak_mapper.viz(maxima=viz_maxima, whh=viz_whh, base=viz_bases)
        # add a time column to each of the peak map tables. Add it via timestep transformation

        contour_line_bounds = (
            # starts with a long frame of 5 columns: p_idx, loc, msnt, dim, value
            peak_map.contour_line_bounds.pipe(pl.from_pandas)
            # pivot on dim with index p_idx, loc, msnt
            .pivot(
                index=[
                    self.keys.P_IDX,
                    self.keys.LOC,
                    self.keys.MSNT,
                ],
                columns=self.keys.DIM,
                values=self.keys.VALUE,
            )
            # add new column 'time' as transform of X_idx_rounded
            .with_columns(
                pl.col(self.keys.IDX_ROUNDED)
                .mul(self._timestep)
                .alias(self.input_keys_time),
            )
            # melt back to long form
            .melt(
                id_vars=[self.keys.P_IDX, self.keys.LOC, self.keys.MSNT],
                variable_name=self.keys.DIM,
                value_name=self.keys.VALUE,
            ).to_pandas()
        )

        # rename "loc" -> "param", transform x axis to time units

        maxima = (
            peak_map.maxima.pipe(pl.from_pandas)
            .pivot(
                index=[self.keys.P_IDX, self.keys.LOC],
                columns=self.keys.DIM,
                values=self.keys.VALUE,
            )
            .select(
                pl.col(self.keys.P_IDX),
                pl.col(self.keys.LOC).alias(self.keys.PARAM),
                pl.col(self.keys.IDX),
                pl.col(self.keys.IDX).mul(self._timestep).alias(self.input_keys_time),
                pl.col(self.keys.X),
            )
            .melt(
                id_vars=[self.keys.P_IDX, self.keys.PARAM],
                variable_name=self.keys.DIM,
                value_name=self.keys.VALUE,
            )
            .to_pandas()
        )

        widths = (
            peak_map.widths.pipe(pl.from_pandas)
            .select(
                pl.col(self.keys.P_IDX),
                pl.col(self.keys.MSNT),
                pl.col(self.keys.VALUE).alias(self.keys.IDX),
                pl.col(self.keys.VALUE).mul(self._timestep).alias(self.input_keys_time),
            )
            .melt(
                id_vars=[self.keys.P_IDX, self.keys.MSNT],
                variable_name=self.keys.UNIT,
                value_name=self.keys.VALUE,
            )
            .to_pandas()
        )

        peak_map = {
            self.keys.CONTOUR_LINE_BOUNDS: contour_line_bounds,
            self.keys.MAXIMA: maxima,
            self.keys.WIDTHS: widths,
        }

        return peak_map, peak_map_plots

    def prepare_maxima_tbl(self, map_peaks_output: pd.DataFrame) -> pl.DataFrame:
        maxima: pl.DataFrame = (
            map_peaks_output[self.keys.MAXIMA].pipe(pl.from_pandas)
            .pivot(index=[self.keys.P_IDX, self.keys.PARAM], columns=self.keys.DIM, values=self.keys.VALUE)
            .rename({
                self.keys.IDX:self.keys.IDX,
            })
            .melt(id_vars=[self.keys.P_IDX,self.keys.PARAM], value_vars=[self.keys.IDX,self.input_keys_time,self.keys.X], variable_name=self.keys.UNIT, value_name=self.keys.VALUE)
            .with_columns(
                pl.when(pl.col(self.keys.UNIT).is_in([self.keys.IDX,self.input_keys_time])).then(pl.lit(self.input_keys_time)).when(pl.col(self.keys.UNIT)==self.keys.X).then(pl.lit(self.keys.Y)).alias(self.keys.DIM)
            )
            .pivot(index=[self.keys.P_IDX,self.keys.PARAM], columns=self.keys.UNIT, values=self.keys.VALUE)
            .with_columns(pl.col(self.keys.IDX).cast(int))
            .melt(
                id_vars=[self.keys.P_IDX,self.keys.IDX,self.keys.PARAM], value_vars=[self.input_keys_time,self.keys.X],variable_name=self.keys.UNIT,value_name=self.keys.VALUE
            )
            .rename({self.keys.PARAM:self.keys.MSNT})
        )  # fmt: skip

        return maxima

    def prepare_tbl_peak_map(
        self,
        maxima,
        widths,
        contour_line_bounds,
    ):

        tbl_peak_map = (
            # join maxima, widths and contour line tables
            pl.concat(
                [
                    maxima.with_columns(
                        pl.col([self.keys.P_IDX]),
                        pl.lit(None).alias(self.keys.SIDE),
                        pl.col([self.keys.MSNT, self.keys.UNIT, self.keys.VALUE]),
                    ),
                    widths.with_columns(
                        pl.col([self.keys.P_IDX]),
                        pl.lit(None).alias(self.keys.SIDE),
                        pl.col([self.keys.MSNT, self.keys.UNIT, self.keys.VALUE]),
                    ),
                    contour_line_bounds,
                ],
                how="diagonal_relaxed",
            )
            # select columns, excluding 'idx'.
            .select(
                self.keys.P_IDX,
                self.keys.MSNT,
                self.keys.SIDE,
                self.keys.UNIT,
                self.keys.VALUE,
            )
            # add window type, idx to peak map
            .pipe(
                lambda df: self.tbl_window_to_peak_idx.select(
                    self.keys.W_TYPE, self.keys.W_IDX, self.keys.P_IDX
                ).join(df, on=self.keys.P_IDX, how="left")
            )
            # add dim column containing the dimension of the measurement, x or y.
            .with_columns(
                pl.when(
                    pl.col(self.keys.UNIT).is_in([self.input_keys_time, self.keys.IDX])
                )
                .then(pl.lit(self.keys.keys_x_unit))
                .when(pl.col(self.keys.UNIT) == self.keys.X)
                .then(pl.lit(self.keys.Y))
                .otherwise(pl.lit(self.keys.UNASSIGNED))
                .alias(self.keys.DIM)
            )
            # want to add a unitcolumn for all x measurements currently in time units. do this by pivoting
            # on unit then adding a new column 'unit_idx' via transformation
            # pivot on 'unit' with index of 'w_type','w_idx','p_idx','msnt',side','dim' and values from 'value'
            .pivot(
                index=[
                    self.keys.W_TYPE,
                    self.keys.W_IDX,
                    self.keys.P_IDX,
                    self.keys.MSNT,
                    self.keys.SIDE,
                    self.keys.DIM,
                ],
                columns=self.keys.UNIT,
                values=self.keys.VALUE,
            )
            # replace the X field with "amp", a synonym
            .rename({self.keys.X: self.keys.AMPLITUDE})
            # add unit_idx column based on transformation from time unit, division by timestep
            .with_columns(
                pl.col(self.input_keys_time)
                .truediv(self._timestep)
                .round(0)
                .cast(int)
                .alias(self.keys.UNIT_IDX)
            )
            # return to long form, excluding X_idx column which contains incorrect values.
            .melt(
                id_vars=[
                    self.keys.W_TYPE,
                    self.keys.W_IDX,
                    self.keys.P_IDX,
                    self.keys.MSNT,
                    self.keys.SIDE,
                    self.keys.DIM,
                ],
                value_vars=[
                    self.input_keys_time,
                    self.keys.AMPLITUDE,
                    self.keys.UNIT_IDX,
                ],
                value_name=self.keys.VALUE,
                variable_name=self.keys.UNIT,
            )
            # the transformations create spurious null cells which need tobe removed
            .filter(~pl.col(self.keys.VALUE).is_null())
            # add new columns
            .with_columns(
                # create a new column: type_ describing the type of the measurement - width, point, contour_line_bound
                pl.when(pl.col(self.keys.MSNT).str.contains(self.keys.WIDTH))
                .then(pl.lit(self.keys.WIDTH))
                .when(pl.col(self.keys.MSNT) == self.keys.MAXIMA)
                .then(pl.lit(self.keys.POINT))
                .when(pl.col(self.keys.SIDE).is_not_null())
                .then(pl.lit(self.keys.CONTOUR_LINE_BOUND))
                .otherwise(pl.lit(self.keys.UNASSIGNED))
                .alias(self.keys.TYPE),
                # within msnt column, remove prefix "width_" as its information is now contained in type_
                pl.col(self.keys.MSNT).str.replace("width_", ""),
            )
        )
        return tbl_peak_map

    def estimate_curve_fit_time(
        self,
        params: pl.DataFrame,
        X_w: pl.DataFrame,
        x_unit: str,
        fit_func: Callable,
        opt_func: Callable,
        schemas: dict[str, Type],
    ):
        """
        Calculate the mean time to complete one iteration of least square minimization of the dataset.

        this is done by calling the curve fit functon 5 times and catching the error thrown by reaching the max_nfev (set to 1).

        The time it takes to throw the error is measured, and the mean calculated.

        returns the mean as a float.
        """

        import time

        elapsed_times = []
        # seems to take a lot longer on the first iteration, I am assuming it compiles the function? run 6
        # times and ignore it from the calculation
        for x in range(6):
            start_time = time.process_time()

            output: dict[str, pl.DataFrame] = self.curve_fit_windows(
                params=params,
                X_w=X_w,
                x_unit=x_unit,
                fit_func=fit_func,
                opt_func=opt_func,
                max_nfev=1,
                n_interms=1,
                schemas=schemas,
                verbose=False,
                terminate_on_fit=False,
            )
            end_time = time.process_time()

            elapsed_times.append(end_time - start_time)

        sample_pool = elapsed_times[1:]

        min_time_stats = {}
        min_time_stats["mean_elapsed_time"] = np.mean(sample_pool)
        min_time_stats["stdev_elapsed_time"] = np.std(sample_pool)

        return min_time_stats


def calculate_timestep(time: ArrayLike):
    timestep = np.mean(np.diff(time))
    return timestep


# def asschrom_dset():
#     path = "tests/test_data/test_assessment_chrom.csv"
#     dset = pd.read_csv(path)
#     return dset


def ringland_dset():
    path = "tests/tests_jonathan/test_data/a0301_2021_chris_ringland_shiraz.csv"

    dset = pl.read_csv(path)

    return dset


def main():
    import panel as pn

    pipeline = DeconvolutionPipeline()

    pipeline.run(
        data=ringland_dset().to_pandas(),
        key_time="time",
        key_amp="signal",
    )


if __name__ == "__main__":
    main()
