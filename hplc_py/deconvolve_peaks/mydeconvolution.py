
"""
    The overall logic of the deconvolution module is as follows:

    1. iterating through each peak window:
        1. Iterating through each peak:
            1. build initial guesses as:
                - amplitude: the peak maxima,
                - location: peak time_idx,
                - width: peak width,
                - skew: 0
            2. build default bounds as:
                - amplitude: 10% peak maxima, 1000% * peak maxima.
                - location: peak window time min, peak window time max
                - width: the timestep, half the width of the window
                - skew: between negative and positive infinity
            3. add custom bounds
            4. add peak specific bounds
        5. submit extracted values to `curve_fit`
        ...

    so we could construct new tables which consist of the initial guesses, upper bounds and lower bounds for each peak in each window, i.e.:

    | # |     table_name   | window | peak | amplitude | location | width | skew |
    | 0 |  initial guesses |   1    |   1  |     70    |    200   |   10  |   0  |

    | # | table_name | window | peak | bound |  amplitude | location | width | skew |
    | 0 |    bounds  |    1   |   1  |   lb  |      7     |    100   | 0.009 | -inf |
    | 1 |    bounds  |    1   |   1  |   ub  |     700    |    300   |  100  | +inf |

    and go from there.

    The initial guess table needs the peak idx to be labelled with windows. since they both ahve the time index, thats fine. we also need the amplitudes from signal df.

    2023-12-08 10:16:41

    This test class now contains methods pertaining to the preparation stage of the deconvolution process.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pt
import tqdm
from pandera.typing import DataFrame
from scipy import optimize
from scipy.optimize._lsq.common import in_bounds

from hplc_py.hplc_py_typing.hplc_py_typing import (
    FloatArray,
    Bounds,
    InitGuesses,
    OutParamsBase,
    OutPeakReportBase,
    Popt,
    Recon,
    OutWindowDF_Base,
    SignalDF,
)

from hplc_py.map_signals.map_windows import WindowedSignal
from hplc_py.map_signals.map_peaks import PeakMap
from hplc_py.skewnorms.skewnorms import SkewNorms

class WdwPeakMap(PeakMap):
    window_type: pd.StringDtype
    window_idx: pd.Int64Dtype
    
    class Config:
        ordered=False
        strict=True
    
@dataclass
class DataPrepper:
    
    ws_sc = WindowedSignal
    pm_sc = PeakMap
    wpm_sc = WdwPeakMap
    
    def try_schema(
        self,
        df: DataFrame|pd.DataFrame,
        schema,
    ):
        df_ = pd.DataFrame()
        
        try:
            df_ = schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as err:
            err.add_note(err.failure_cases.to_markdown())
            err.add_note("\n\n")
            err.add_note(err.data.to_markdown())
            raise err
        
        return df_
    
    """
    class containing methods to prepare the data for deconvolution
    """

    def _window_peak_map(
        self,
        pm: DataFrame[PeakMap],
        ws: DataFrame[WindowedSignal],
    )-> DataFrame[WdwPeakMap]:
        
        wpm_ = pm.copy(deep=True)
        ws_ = ws.copy(deep=True)
        
        wpm_ = wpm_.reset_index().set_index(self.pm_sc.time_idx)
        
        ws_ = ws_.drop([self.ws_sc.time,self.ws_sc.amp],axis=1).set_index(self.ws_sc.time_idx)
        
        wpm_ = wpm_.join(ws_, how='left', validate="1:1").reset_index()
        
        wpm_idx_cols = [
            self.wpm_sc.window_type,
            self.wpm_sc.window_idx,
            self.wpm_sc.p_idx,
            self.wpm_sc.time_idx,
            self.wpm_sc.time,
            ]
        
        wpm_ = wpm_.set_index(wpm_idx_cols).reset_index().set_index(self.wpm_sc.idx)
        
        wpm = self.try_schema(wpm_, self.wpm_sc)
        
        return wpm
    
    @pa.check_types
    def _p0_factory(
        self,
        pm: DataFrame[PeakMap],
        ws: DataFrame[WindowedSignal],
    ) -> pt.DataFrame[InitGuesses]:
        """
        Build a table of initial guesses for each peak in each window following the format:

            | # |     table_name   | window | peak_idx | amplitude | location | width | skew |
            | 0 |  initial guesses |   1    |   1  |     70    |    200   |   10  |   0  |

        The initial guess for each peak is simply its maximal amplitude, the time idx of the maximal amplitude, estimated width divided by 2, and a skew of zero.
        """
        # window the peak map
        
        p0 = pd.DataFrame()

        # enforce rational ordering of columns
        p0 = p0.reindex(
            ["window_idx", "peak_idx", "time", "amp_corrected", "whh"], axis=1
        )

        # test for unexpected NA
        if (p0.isna()).any().any():
            error_str = "NA detected:"
            na_rows = p0.isna().index
            nas = p0.loc[na_rows, :]
            raise ValueError(error_str + "\n\n" + str(nas))

        # rename cols to match my definitions

        p0 = p0.rename({"time": "loc", int_col: "amp"}, axis=1, errors="raise")

        # assign skew as zero as per definition
        p0["skew"] = 0

        # assign whh as half peak whh as per definition, in time units

        p0["whh"] = p0["whh"].transform(lambda x: x / 2 * timestep)

        # melt frame to get p0 values as 1 column with param label column for each row
        p0 = p0.melt(
            id_vars=["window_idx", "peak_idx"],
            value_vars=["loc", "amp", "whh", "skew"],
            value_name="p0",
            var_name="param",
        )

        # test for unexpected NA
        if (p0.isna()).any().any():
            error_str = "NA detected:"
            na_rows = p0.isna().index
            nas = p0.loc[na_rows, :]
            raise ValueError(error_str + "\n\n" + str(nas))

        # set param label column as ordered categorical
        p0 = (
            p0.astype(
                {
                    "param": pd.CategoricalDtype(
                        ["amp", "loc", "whh", "skew"], ordered=True
                    ),
                    "window_idx": pd.Int64Dtype(),
                    "peak_idx": pd.Int64Dtype(),
                    "p0": pd.Float64Dtype(),
                }
            )
            .sort_values(by=["window_idx", "peak_idx", "param"])
            .reset_index(drop=True)
        )

        if (p0.isna()).any().any():
            error_str = "NA detected:"
            na_rows = p0.isna().index
            nas = p0.loc[na_rows, :]
            raise ValueError(error_str + "\n\n" + str(nas))

        return pt.DataFrame[InitGuesses](p0)

    @pa.check_types
    def _default_bounds_factory(
        self,
        p0_df: pt.DataFrame[InitGuesses],
        signal_df: pt.DataFrame[SignalDF],
        window_df: pt.DataFrame[OutWindowDF_Base],
        peak_df: pt.DataFrame[PeakMap],
        timestep: np.float64,
    ) -> pt.DataFrame[Bounds]:
        """
        Build a default bounds df from the `signal_df`, `peak_df`, and `window_df` in the following format:

        | # | table_name | window | peak_idx | bound |  amp | location | width | skew |
        |---|------------|--------|----------|-------|------|----------|-------|------|
        | 0 |    bounds  |    1   |     1    |   lb  |   7  |    100   | 0.009 | -inf |
        | 1 |    bounds  |    1   |     1    |   ub  |  70  |    300   |  100  | +inf |

        The bounds are defined as follows:

        | parameter | bound | formula                          |
        |-----------|-------|----------------------------------|
        | amplitude |   lb  | 10% peak maxima                  |
        | amplitude |   ub  | 1000% peak maxima                |
        | location  |   lb  | minimum time index of the window |
        | location  |   ub  | maximum time index of the window |
        | width     |   lb  | magnitude of the timestep        |
        | width     |   ub  | half the width of the window     |
        | skew      |   lb  | -inf                             |
        | skew      |   ub  | +inf                             |

        - amplitude: 10% peak maxima, 1000% * peak maxima.
                - location: peak window time min, peak window time max
                - width: the timestep, half the width of the window
                - skew: between negative and positive infinity
        """

        timestep = np.float64(timestep)

        # construct a container df for the bounds. needs to have the window_idx, peak_id, then bounds as [ub, lb]

        # join the window and peak dfs on time_idx, left join on peak_df

        """
        |    |   window_idx |   peak_idx | param   |   p0 |
        |---:|------------:|-----------:|:--------|----------------:|
        |  0 |           1 |          0 | amp     |        42.6901  |
        |  4 |           1 |          1 | amp     |        19.9608  |
        |  8 |           1 |          2 | amp     |         2.65962 |
        | 12 |           2 |          3 | amp     |        39.8942  |
        """

        amp = (
            p0_df.query("param=='amp'")
            .copy(deep=True)
            .loc[:, ["window_idx", "peak_idx", "param", "p0"]]
            .assign(lb=lambda df: df["p0"] * 0.1)
            .assign(ub=lambda df: df["p0"] * 10)
            .drop(["p0"], axis=1)
        )
        # location

        loc = self._get_loc_bounds(
            signal_df,
            peak_df,
            window_df,
        )

        # width

        # lb is the timestep, just assign directly. ub is half the window width.
        # window width is (window_max-window_min)/2, can get that from the loc bounds

        whh = (
            loc.copy(deep=True)
            .assign(param="whh")
            .assign(whh_lb=timestep)
            .assign(whh_ub=lambda df: (df["ub"] - df["lb"]) / 2)
            .drop(["lb", "ub"], axis=1)
            .rename({"whh_ub": "ub", "whh_lb": "lb"}, axis=1, errors="raise")
        )

        # skew bounds are defined as anywhere between negative and positive infinity
        skew = (
            peak_df.copy(deep=True)
            .loc[:, ["time_idx", "peak_idx"]]
            .set_index("time_idx")
            .join(
                window_df.loc[:, ["time_idx", "window_idx"]].set_index("time_idx"),
                how="left",
            )
            .reset_index(drop=True)
            .assign(lb=-np.inf)
            .assign(ub=np.inf)
        )
        skew.insert(2, "param", "skew")

        bounds = (
            pd.concat([amp, loc, whh, skew])
            .sort_values(["window_idx", "peak_idx", "param"])
            # define param as a categorical for ordering
            .astype(
                {
                    "param": pd.CategoricalDtype(
                        ["amp", "loc", "whh", "skew"], ordered=True
                    )
                }
            )
            .reset_index(drop=True)
        )

        return DataFrame[Bounds](bounds)

    def _get_loc_bounds(
        self,
        signal_df: pt.DataFrame[SignalDF],
        peak_df: pt.DataFrame[PeakMap],
        window_df: pt.DataFrame[OutWindowDF_Base],
    ):
        """
        Define the upper and lower bounds of the time domain of each peak as the extremes of the window each peak is assigned to. This is achieved by as series of joins and pivots to first label each peak then summarise the range of each window and combine the two. Returns two dataframes containing the labelled upper and lower bound series
        'window_max' and'window_min', respectively.
        """

        # get the window each peak belongs to
        peak_df_window_df = (
            peak_df.set_index("time_idx")
            .loc[:, ["peak_idx"]]
            .join(
                [
                    window_df.query("window_type=='peak'")
                    .set_index("time_idx")
                    .loc[:, "window_idx"],
                ]
            )
            .reset_index()
        )

        # display the min and max of each window

        # test if join worked

        if peak_df_window_df.isna().any().any():
            raise ValueError("Unexpected behavior - NA present after join")
        # get time index bounds of each window

        pivot_window_df = window_df.query("window_type=='peak'").pivot_table(
            index="window_idx", values="time_idx", aggfunc=["min", "max"]
        )

        pivot_window_df = pivot_window_df.rename_axis(
            columns=["bound", "time_unit"]
        ).sort_index(axis=1)

        stack_window_df = pivot_window_df.stack("bound").reset_index()

        # join with signal_df to get time values
        join_df = (
            stack_window_df.set_index("time_idx")
            .join(signal_df.set_index("time_idx").loc[:, "time"], how="left")
            .reset_index()
        )
        # join with peak_df_window_df to assign the bound to each peak

        interm_tbl = (
            join_df.set_index("window_idx")
            .join(
                peak_df_window_df.set_index("window_idx").loc[:, ["peak_idx"]],
                how="left",
            )
            .reset_index()
            .sort_values(["window_idx", "peak_idx"])
        )

        # pivot such that columns become 'lb','ub'
        loc_bounds = (
            interm_tbl.pivot_table(
                values="time", columns="bound", index=["window_idx", "peak_idx"]
            )
            .reset_index()
            .rename({"min": "lb", "max": "ub"}, axis=1, errors="raise")
        )

        # add 'param' label col
        loc_bounds["param"] = "loc"

        # reorder the columns to hint at the index cols
        loc_bounds = loc_bounds.reindex(
            ["window_idx", "peak_idx", "param", "lb", "ub"], axis=1
        )

        # set dtypes
        loc_bounds = loc_bounds.astype(
            {
                "window_idx": pd.Int64Dtype(),
                "peak_idx": pd.Int64Dtype(),
                "param": pd.StringDtype(),
                "lb": pd.Float64Dtype(),
                "ub": pd.Float64Dtype(),
            }
        )

        return loc_bounds

    @pa.check_types
    def _param_df_factory(
        self,
        p0: pt.DataFrame[InitGuesses],
        default_bounds: pt.DataFrame[Bounds],
    ) -> pt.DataFrame[OutParamsBase]:
        """
        combine the p0 and default_bounds df to allow for easy
        comparison between the three values for each parameter. returns the
        combined df
        """

        # join p0 and bounds

        param_df = (
            p0.set_index(["window_idx", "peak_idx", "param"])
            .join(
                default_bounds.set_index(["window_idx", "peak_idx", "param"]),
                how="left",
            )
            .reset_index()
        )

        # test whether NA present

        if param_df.isna().sum().sum() > 0:
            raise ValueError(
                f"NA present in `param_df`:\n{param_df[param_df.isna().any(axis=1)]}"
            )

        # add a test for oob

        param_df = param_df.assign(
            inbounds=lambda df: df.apply(
                lambda x: in_bounds(x["p0"], x["lb"], x["ub"]),
                axis=1,
            )
        )

        if not param_df["inbounds"].all():
            raise ValueError(
                "oob guess detected" f"\n{param_df.query('inbounds==False')}"
            )

        return pt.DataFrame[OutParamsBase](param_df)

    def _prepare_data(
        self,
        signal_df: pd.DataFrame,
        peak_df: pd.DataFrame,
        window_df: pd.DataFrame,
        timestep: float,
    ):
        p0_df = self.dataprepper.p0_factory(
            signal_df,
            peak_df,
            window_df,
            timestep,
            "amp_corrected",
        )

        default_bounds = self.dataprepper.default_bounds_factory(
            p0_df,
            signal_df,
            window_df,
            peak_df,
            timestep,
        )
        param_df = self.dataprepper._param_df_factory(
            p0_df,
            default_bounds,
        )

        windowed_signal_df = self.dataprepper._window_signal_df(signal_df, window_df)

        return p0_df, default_bounds, param_df, windowed_signal_df


@dataclass
class PeakDeconvolver(SkewNorms):
    """
    Note: as of 2023-12-08 15:10:06 it is necessary to inherit SkewNorms rather than passing
    class method objects in the `curve_fit` call because `_fit_skewnorms` is defined with the
    packing operator for 'params' rather than explicit arguments. This is causing unexpected
    behavior where the arguments 'xdata', 'p0', etc are bumped up one in the unpacking, resulting in
    the xdata value being assigned to the 'self' parameter, the params[0] (amp) being assigned to x, etc.

    Do not have time to solve this problem as it boils down to a paradigm choice rather than a critical feature,
    thus it will be left for a later date when I can conclude whether the pack operator is necessary, i.e. due to
    the behavior of `curve_fit`, and whether there is another option, i.e. excluding self somehow.

    """

    dataprepper = DataPrepper()

    @pa.check_types
    def deconvolve_peaks(
        self,
        ws: DataFrame[WindowedSignal],
    ) -> tuple[pt.DataFrame[Popt], pt.DataFrame[Recon]]:
        
        p0_df, default_bounds, param_df, windowed_signal_df = self.dataprepper.prepare_data(
            signal_df, peak_df, window_df, timestep
        )

        self._param_df = param_df

        popt_df: pd.DataFrame = self._popt_factory(windowed_signal_df, param_df)

        reconstructed_signals = self._reconstruct_peak_signal(
            windowed_signal_df["time"].to_numpy(np.float64), popt_df
        )

        return popt_df.pipe(pt.DataFrame[Popt]), reconstructed_signals.pipe(
            pt.DataFrame[Recon]
        )  # type: ignore

    def _get_peak_report(
        self,
        popt_df: pt.DataFrame[Popt],
        unmixed_df: pt.DataFrame[Recon],
        timestep: np.float64,
    ):
        """
        add peak area to popt_df. Peak area is defined as the sum of the amplitude arrays
        of each peak
        """

        # groupby peak idx and calculate the area as the sum of amplitudes, and the maxima
        # mst - measurement
        unmixed_mst = (
            unmixed_df.groupby("peak_idx")["unmixed_amp"]
            .agg(["sum", "max"])
            .rename(
                {"sum": "unmixed_area", "max": "unmixed_maxima"}, axis=1, errors="raise"
            )
        )

        peak_report_df = (
            popt_df.copy(deep=True)
            .set_index("peak_idx")
            .join(unmixed_mst)
            .reset_index()
            .assign(tbl_name="peak_report")
            # express loc as retention time
            .assign(retention_time=lambda df: df["loc"] * timestep)
            .astype(
                {
                    "tbl_name": pd.StringDtype(),
                    "retention_time": pd.Float64Dtype(),
                }
            )
            .loc[
                :,
                [
                    "tbl_name",
                    "window_idx",
                    "peak_idx",
                    "retention_time",
                    "loc",
                    "amp",
                    "whh",
                    "skew",
                    "unmixed_area",
                    "unmixed_maxima",
                ],
            ]
        )
        # assert False, "\n"+"\n".join(peak_report_df.columns)

        return peak_report_df.pipe(pt.DataFrame[OutPeakReportBase])

    def _prep_for_curve_fit(
        self,
        window: int,
        ws_df: pt.DataFrame[WindowedSignal],
        amp_col: str,
        param_df: pt.DataFrame[OutParamsBase],
    ):
        """ """

        x: pd.Series[float] = ws_df.loc[ws_df["window_idx"] == window, "time"].astype(
            np.float64
        )

        y: pd.Series[float] = ws_df.loc[ws_df["window_idx"] == window, amp_col].astype(
            np.float64
        )

        p0: pd.DataFrame = param_df.loc[ws_df["window_idx"] == window]
        p0: pd.DataFrame = p0.set_index(["window_idx", "peak_idx", "param"])
        p0: pd.Series[float] = p0.loc[:, "p0"]

        lb: pd.Series[float] = param_df.loc[ws_df["window_idx"] == window].set_index(
            "param"
        )["lb"]
        ub: pd.Series[float] = param_df.loc[ws_df["window_idx"] == window].set_index(
            "param"
        )["ub"]

        # input validation

        if not x.ndim == 1:
            raise ValueError("ndim x should be 1")
        if not y.ndim == 1:
            raise ValueError("ndim y should be 1")
        if not p0.ndim == 1:
            raise ValueError("ndim p0 should be 1")
        if not lb.ndim == 1:
            raise ValueError("ndim lb should be 1")
        if not ub.ndim == 1:
            raise ValueError("ndim ub should be 1")
        # assert False, f"\n{len(p0)}"

        if len(x) == 0:
            raise ValueError("len x must be greater than zero")
        if len(y) == 0:
            raise ValueError("len y must be greater than zero")
        if len(p0) == 0:
            raise ValueError("len p0 must be greater than zero")
        if len(lb) == 0:
            raise ValueError("len lb must be greater than zero")
        if len(ub) == 0:
            raise ValueError("len ub must be greater than zero")

        """
            From the docs: `curve_fit` returns an array of "Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized."
            """
        # check that each array is the appropriate length divisible by 4

        if len(p0) % 4 != 0:
            raise ValueError("length of p0 is not a multiple of 4")
        if len(lb) % 4 != 0:
            raise ValueError("length of lb is not a multiple of 4")
        if len(ub) % 4 != 0:
            raise ValueError("length of ub is not a multiple of 4")

        return x, y, p0, lb, ub

    def _popt_factory(
        self,
        windowed_signal_df: pt.DataFrame[WindowedSignal],
        param_df: pt.DataFrame[OutParamsBase],
        verbose=True,
        optimizer_kwargs={},
    ) -> pt.DataFrame[Popt]:
        popt_list = []

        windows = windowed_signal_df.loc[lambda df: df["window_type"] == "peak"][
            "window_idx"
        ].unique()

        if verbose:
            windows_itr = tqdm.tqdm(windows, desc="deconvolving windows")

        else:
            windows_itr = windows

        for window in windows_itr:
            x, y, p0, lb, ub = self._prep_for_curve_fit(
                window, windowed_signal_df, "amp_corrected", param_df
            )

            (
                popt,
                _,
                infodict,
                mesg,
                ier,
            ) = optimize.curve_fit(
                self._fit_skewnorms,
                xdata=x.to_numpy(np.float64),
                ydata=y.to_numpy(np.float64),
                p0=p0.to_numpy(np.float64),
                bounds=(lb.to_numpy(np.float64), ub.to_numpy(np.float64)),
                # maxfev=100,
                full_output=True,
                **optimizer_kwargs,
            )

            # the output of `curve_fit` appears to not match the input ordering. Could

            popt_series = pd.Series(
                popt, index=p0.index, dtype=pd.Float64Dtype(), name="popt"
            )

            popt_list.append(popt_series)

        popt_df = (
            pd.concat(popt_list)
            .reset_index()
            .pivot_table(
                columns="param", index=["window_idx", "peak_idx"], values="popt"
            )
            .reset_index()
            .assign(
                tbl_name=lambda df: pd.Series(
                    ["popt"] * len(df), dtype=pd.StringDtype()
                )
            )
            .loc[:, ["tbl_name", "window_idx", "peak_idx", "amp", "loc", "whh", "skew"]]
            .rename_axis(index="idx", columns="cols")
        )

        return pt.DataFrame[Popt](popt_df)

    # @pa.check_types
    def _reconstruct_peak_signal(
        self,
        time: FloatArray,
        popt_df: pt.DataFrame[Popt],
    ) -> pt.DataFrame[Recon]:
        def reconstruct_peak_signal(
            popt_df: pt.DataFrame[Popt],
            time: pt.Series,
        ) -> pd.DataFrame:
            params = popt_df.loc[:, ["amp", "loc", "whh", "skew"]].values.flatten()

            unmixed_signal = self._compute_skewnorm(time.values, *params)

            unmixed_signal_df = pd.merge(
                popt_df.loc[:, ["peak_idx"]], time, how="cross"
            )

            unmixed_signal_df = unmixed_signal_df.assign(unmixed_amp=unmixed_signal)

            unmixed_signal_df = unmixed_signal_df.reset_index(names="time_idx")
            unmixed_signal_df["time_idx"] = unmixed_signal_df["time_idx"].astype(
                pd.Int64Dtype()
            )
            unmixed_signal_df = unmixed_signal_df.loc[
                :, ["peak_idx", "time_idx", "time", "unmixed_amp"]
            ]

            unmixed_signal_df = unmixed_signal_df.reset_index(drop=True)

            return unmixed_signal_df

        # remove window_idx from identifier to avoid multiindexed column

        unmixed_df = popt_df.groupby(
            by=["peak_idx"],
            group_keys=False,
        ).apply(reconstruct_peak_signal, *[time])

        unmixed_df = unmixed_df.reset_index(drop=True)

        return DataFrame[Recon](unmixed_df)

    def _optimize_p(
        self,
        p0,
        default_bounds,
        signal_df: pt.DataFrame[SignalDF],
        window_df: pt.DataFrame[OutWindowDF_Base],
    ) -> pd.DataFrame:
        windowed_signal_df = self._window_signal_df(
            signal_df,
            window_df,
        )

        # join the initial guess and default bound tables

        param_df = self._param_df_factory(
            p0,
            default_bounds,
        )

        popt_df = self._popt_factory(
            windowed_signal_df,
            param_df,
        )

        return popt_df
