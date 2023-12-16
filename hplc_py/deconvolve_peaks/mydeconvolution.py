import matplotlib.pyplot as plt

from collections import namedtuple

import pandas as pd
import pandera as pa
import pandera.typing as pt

import numpy as np
import numpy.typing as npt
from scipy import optimize
from scipy.optimize._lsq.common import in_bounds

import warnings
import tqdm

from hplc_py.skewnorms import skewnorms
from hplc_py.deconvolve_peaks import windowstate
from hplc_py.skewnorms.skewnorms import SkewNorms
from hplc_py.hplc_py_typing.hplc_py_typing import (
    OutSignalDF_Base,
    OutPeakDF_Base,
    OutWindowDF_Base,
    OutInitialGuessBase,
    OutDefaultBoundsBase,
    OutWindowedSignalBase,
    OutParamsBase,
    OutPoptBase,
    OutReconDFBase,
    OutPeakReportBase,
)


class DataPrepper(skewnorms.SkewNorms):
    """
    class containing methods to prepare the data for deconvolution
    """

    def find_integration_range(
        self,
        time_idx: pt.Series[np.int64],
        integration_window: list[float],
    ) -> npt.NDArray[np.float64]:
        t_range = 0

        # Determine the areas over which to integrate the window
        if len(integration_window) == 0:
            t_range = time_idx.values

        elif type(integration_window) == list:
            if len(integration_window) == 2:
                t_range = time_idx.loc[integration_window[0] : integration_window[1]]
            else:
                raise RuntimeError(
                    "Provided integration bounds has wrong dimensions. Should have a length of 2."
                )

        t_range = np.asarray(t_range, np.float64)

        return t_range

    def p0_factory(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
    ):
        """
        Build a table of initial guesses for each peak in each window following the format:

            | # |     table_name   | window | peak_idx | amplitude | location | width | skew |
            | 0 |  initial guesses |   1    |   1  |     70    |    200   |   10  |   0  |

        The initial guess for each peak is simply its maximal amplitude, the time idx of the maximal amplitude, estimated width divided by 2, and a skew of zero.
        """
        # join the tables with two left joins on peak_df.

        OutSignalDF_Base(signal_df)
        OutPeakDF_Base(peak_df)
        OutWindowDF_Base(window_df)

        p0_df = (
            peak_df.copy(deep=True)
            .rename({"time_idx": "loc"}, axis=1)
            .loc[:, ["peak_idx", "loc", "whh"]]
            .set_index("loc")
            .join(
                [
                    signal_df.loc[:, ["amp"]],
                    window_df.loc[:, ["time_idx", "window_idx"]].set_index("time_idx"),
                ],
                how="left",
                validate="1:1",
            )
            .assign(skew=0)
            .assign(whh=lambda df: df['whh']/2)
            .reset_index()
            .set_index(["window_idx", "peak_idx"])
            .melt(
                ignore_index=False,
                value_name="p0",
                var_name="param",
            )
            .reset_index()
            .astype(
                {
                    "param": pd.CategoricalDtype(
                        ["amp", "loc", "whh", "skew"], ordered=True
                    ),
                    "window_idx": pd.Int64Dtype(),
                    "peak_idx": pd.Int64Dtype(),
                }
            )
            .sort_values(by=["window_idx", "peak_idx", "param"])
            .reset_index(drop=True)
        )

        return pt.DataFrame[OutInitialGuessBase](p0_df)

    def default_bounds_factory(
        self,
        p0_df: pt.DataFrame[OutInitialGuessBase],
        window_df: pt.DataFrame[OutWindowDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        timestep: np.float64,
    ) -> pt.DataFrame[OutDefaultBoundsBase]:
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

        # input validation
        pt.DataFrame[OutInitialGuessBase](p0_df)
        pt.DataFrame[OutWindowDF_Base](window_df)
        pt.DataFrame[OutPeakDF_Base](peak_df)

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

        loc = self.get_loc_bounds(
            peak_df,
            window_df,
        )

        # width

        # lb is the timestep, just assign directly. ub is half the window width.
        # window width is (window_max-window_min)/2, can get that from the loc bounds
        
        whh = (
            loc.copy(deep=True)
            .assign(param="whh")
            .assign(whh_lb=1)
            .assign(whh_ub=lambda df: (df["ub"] - df["lb"]) / 2)
            .drop(["lb", "ub"], axis=1)
            .rename({"whh_ub": "ub", "whh_lb": "lb"}, axis=1)
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
            .astype(
                {
                    "param": pd.CategoricalDtype(
                        ["amp", "loc", "whh", "skew"], ordered=True
                    )
                }
            )
            .reset_index(drop=True)
        )

        # define param as a categorical for ordering

        return bounds

    def get_loc_bounds(
        self,
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
    ):
        """
        Define the upper and lower bounds of the time domain of each peak as the extremes of the window each peak is assigned to. This is achieved by as series of joins and pivots to first label each peak then summarise the range of each window and combine the two. Returns two dataframes containing the labelled upper and lower bound series
        'window_max' and'window_min', respectively.
        """
        peak_df_window_df = (
            peak_df.loc[:, ["time_idx", "peak_idx"]]
            .set_index(["time_idx"])
            .join(window_df.loc[:, ["time_idx", "window_idx"]].set_index(["time_idx"]))
            .reset_index()
        )

        # construct a table who has the window id as the index and the min and max time values of each window as columns

        pivot_window_df = (
            window_df.pivot_table(
                columns="window_idx", values="time_idx", aggfunc=["min", "max"]
            )
            .pipe(lambda df: df.set_axis(df.columns.set_names("agg", level=0), axis=1))
            .melt(value_name="time_idx")
            .pivot_table(values="time_idx", columns="agg", index="window_idx")
            .loc[:, ["min", "max"]]
            .rename({"min": "lb", "max": "ub"}, axis=1)
        )

        # left join the tables on window_idx

        loc_bounds = (
            peak_df_window_df.loc[:, ["window_idx", "peak_idx"]]
            .set_index("window_idx")
            .join(
                pivot_window_df,
                how="left",
            )
            .reset_index()
        )

        loc_bounds.insert(2, "param", "loc")

        return loc_bounds

    def _window_signal_df(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
    ) -> pt.DataFrame[OutWindowedSignalBase]:
        """
        Create a windowed version of the signal df for subsetting
        """
        windowed_signal_df = (
            signal_df.join(
                window_df.loc[:, ["time_idx", "window_idx"]].set_index(["time_idx"]),
                how="left",
            )
            .dropna()
            .astype({"window_idx": pd.Int64Dtype()})
        )
        return windowed_signal_df

    def _param_df_factory(
        self,
        p0: pt.DataFrame[OutInitialGuessBase],
        default_bounds: pt.DataFrame[OutDefaultBoundsBase],
    ) -> pt.DataFrame[OutParamsBase]:
        """
        combine the p0 and default_bounds df to allow for easy
        comparison between the three values for each parameter. returns the
        combined df
        """

        # input validation

        OutInitialGuessBase(p0)
        OutDefaultBoundsBase(default_bounds)

        param_df = (
            p0.set_index(["window_idx", "peak_idx", "param"])
            .join(
                default_bounds.set_index(["window_idx", "peak_idx", "param"]), how="left"
            )
            .reset_index()
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
                "oob guess detected" f"{param_df.query('inbounds==False')}"
            )
        return param_df.pipe(pt.DataFrame[OutParamsBase])  # type: ignore


class PPeakDeconvolver(SkewNorms):
    """
    Note: as of 2023-12-08 15:10:06 it is necessary to inherit SkewNorms rather than passing
    class method objects in the `curve_fit` call because `_fit_skewnorms` is defined with the
    packing operator for 'params' rather than explicit arguments. This is causing unexpected
    behavior where the arguments 'xdata', 'p0', etc are bumped up one in the unpacking, resulting in
    the xdata value being assigned to the 'self' parameter, the params[0] (amp) being assigned to x, etc.

    Do not have time to solve this problem as it boils down to a paradigm choice rather than a critical feature,
    thus it will be left for a later date when I can conclude whether the pack operator is necessary, i.e. due to
    the behavior of `curve_fit`, and whether there is another option, i.e. excluding self somehow.

    TODO:
    - [x] establish master deconvolution method
    - [x] add area to popt_df returning as 'peak_params'
    - [ ] refactor assess fit methods to assess performance of my code
    - [ ] run dataset on original code to get goodness of fit and compare
    - [ ] complete schemas for new tables i.e. `popt_df`.
    - [ ] define peak maxima as the maxima of the reconstructed signal, relabel 'amp' as param_amp or something.
    """

    def __init__(self):
        self.dataprepper = DataPrepper()

    def deconvolve_peaks(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
        timestep: np.float64,
    ) -> tuple[OutPoptBase, OutReconDFBase]:
        windowed_signal_df = self.dataprepper._window_signal_df(signal_df, window_df)

        p0_df = self.dataprepper.p0_factory(
            signal_df,
            peak_df,
            window_df,
        )

        default_bounds = self.dataprepper.default_bounds_factory(
            p0_df,
            window_df,
            peak_df,
            timestep,
        )
        param_df = self.dataprepper._param_df_factory(
            p0_df,
            default_bounds,
        )

        self._param_df = param_df
        
        popt_df = self._popt_factory(windowed_signal_df, param_df)

        reconstructed_signals = self._reconstruct_peak_signal(
            windowed_signal_df["time_idx"], popt_df
        )

        return popt_df.pipe(pt.DataFrame[OutPoptBase]), reconstructed_signals.pipe(pt.DataFrame[OutReconDFBase])  # type: ignore

    def compile_peak_report(
        self,
        popt_df: pt.DataFrame[OutPoptBase],
        unmixed_df: pt.DataFrame[OutReconDFBase],
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
            .rename({"sum": "unmixed_area", "max": "unmixed_maxima"}, axis=1)
        )

        peak_report_df = (
            popt_df.copy(deep=True)
            .set_index("peak_idx")
            .join(unmixed_mst)
            .reset_index()
            .assign(tbl_name="peak_report")
            # express loc as retention time
            .assign(retention_time=lambda df: df['loc']*timestep)
            .astype({"tbl_name": pd.StringDtype(),
                     "retention_time": pd.Float64Dtype(),
                     })
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
    
        
        return peak_report_df.pipe(pt.DataFrame[OutPeakReportBase])

    def _prep_for_curve_fit(
        self,
        window: int,
        windowed_signal_df: pt.DataFrame[OutWindowedSignalBase],
        param_df: pt.DataFrame[OutParamsBase],
    ):
        """ """
        x = windowed_signal_df.query("window_idx==@window").loc[:, "time_idx"]

        y = windowed_signal_df.query("window_idx==@window").loc[:, "amp"]

        p0 = (
            param_df.query("window_idx==@window")
            .set_index(["window_idx", "peak_idx", "param"])
            .loc[:, "p0"]
        )

        lb = param_df.query("(window_idx==@window)").set_index("param")["lb"]
        ub = param_df.query("(window_idx==@window)").set_index("param")["ub"]

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
        windowed_signal_df: pt.DataFrame[OutWindowedSignalBase],
        param_df: pt.DataFrame[OutParamsBase],
        verbose=True,
        optimizer_kwargs={},
    )->pt.DataFrame[OutPoptBase]:
        popt_list = []

        windows = windowed_signal_df["window_idx"].unique()
        
        if verbose:
            windows_itr = tqdm.tqdm(windows, desc="deconvolving windows")

        else:
            windows_itr = windows

        for window in windows_itr:
            x, y, p0, lb, ub = self._prep_for_curve_fit(
                window, windowed_signal_df, param_df
            )

            popt, _, infodict, mesg, ier, = optimize.curve_fit(
                self._fit_skewnorms,
                xdata=x.to_numpy(np.float64),
                ydata=y.to_numpy(np.float64),
                p0=p0.to_numpy(np.float64),
                bounds=(lb.to_numpy(np.float64), ub.to_numpy(np.float64)),
                maxfev=100,
                full_output=True,
                **optimizer_kwargs,
            )

            print(mesg)
            print(infodict)
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

        return popt_df

    def _reconstruct_peak_signal(
        self,
        xdata,
        popt_df: pd.DataFrame,
    ):
        def reconstruct_peak_signal(
            popt_df,
            xdata,
        ):
            params = popt_df.loc[:, ["amp", "loc", "whh", "skew"]].values.flatten()
            
            try:
                unmixed_signal = self._compute_skewnorm(xdata.values, *params)

                unmixed_signal_df = pd.merge(popt_df.loc[:, ["peak_idx"]], xdata, how="cross")

                unmixed_signal_df = unmixed_signal_df.assign(unmixed_amp=unmixed_signal)

                return unmixed_signal_df

            except Exception as e:
                raise RuntimeError(f"{e}\n" f"{params}\n" f"{popt_df.index}\n")

        # remove window_idx from identifier to avoid multiindexed column
         
        unmixed_df = popt_df.groupby(
            by=["peak_idx"],
            group_keys=False,
        ).apply(
            reconstruct_peak_signal, xdata
        )  # type: ignore

        return unmixed_df

    def optimize_p(
        self,
        p0,
        default_bounds,
        signal_df: pt.DataFrame[OutSignalDF_Base],
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