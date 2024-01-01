from typing import Annotated

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
    OutPoptDF_Base,
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

    @pa.check_types
    def p0_factory(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
        timestep: np.float64,
        int_col: str,
    ) -> pt.DataFrame[OutInitialGuessBase]:
        """
        Build a table of initial guesses for each peak in each window following the format:

            | # |     table_name   | window | peak_idx | amplitude | location | width | skew |
            | 0 |  initial guesses |   1    |   1  |     70    |    200   |   10  |   0  |

        The initial guess for each peak is simply its maximal amplitude, the time idx of the maximal amplitude, estimated width divided by 2, and a skew of zero.
        """
        # join the tables with two left joins on peak_df.

        p0 = peak_df.copy(deep=True)
        
        p0 = p0.reindex(['peak_idx','time_idx','whh'],axis=1)
        
        # get the time values based on their idx
        
        # test for unexpected NA
        if (p0.isna()).any().any():
                error_str = "NA detected:"
                na_rows = p0.isna().index
                nas = p0.loc[na_rows,:]
                raise ValueError(error_str+"\n\n"+str(nas))
        
        # get the time, amplitudes, window idxs from signal_df and window_df
        p0 = (
        p0
        .set_index("time_idx")
        .join(
            [
                signal_df.set_index('time_idx').loc[:, ['time',int_col]],
                window_df.set_index("time_idx").loc[:, "window_idx"],
            ],
            how="left",
            validate="1:1",
        )
        .reset_index()
        )
        
        # enforce rational ordering of columns
        p0 = p0.reindex(['window_idx','peak_idx','time','amp_corrected','whh'], axis=1)
        
        # test for unexpected NA
        if (p0.isna()).any().any():
            error_str = "NA detected:"
            na_rows = p0.isna().index
            nas = p0.loc[na_rows,:]
            raise ValueError(error_str+"\n\n"+str(nas))

        # rename cols to match my definitions
        
        p0 = p0.rename({"time": "loc",
                        int_col:"amp"}, axis=1, errors="raise")
        
        # assign skew as zero as per definition
        p0['skew']=0
        
        # assign whh as half peak whh as per definition, in time units
        p0['whh']=p0['whh']/2*timestep
        
        # melt frame to get p0 values as 1 column with param label column for each row
        p0 = (
            p0
            .melt(
                id_vars=['window_idx','peak_idx'],
                value_vars=['loc','amp','whh','skew'],
                value_name="p0",
                var_name="param",
            )
        )

        # test for unexpected NA
        if (p0.isna()).any().any():
            error_str = "NA detected:"
            na_rows = p0.isna().index
            nas = p0.loc[na_rows,:]
            raise ValueError(error_str+"\n\n"+str(nas))
        
        # set param label column as ordered categorical
        p0 = (
            p0.astype(
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

        if (p0.isna()).any().any():
            error_str = "NA detected:"
            na_rows = p0.isna().index
            nas = p0.loc[na_rows,:]
            raise ValueError(error_str+"\n\n"+str(nas))
                
        return p0

    @pa.check_types
    def default_bounds_factory(
        self,
        p0_df: pt.DataFrame[OutInitialGuessBase],
        signal_df: pt.DataFrame[OutSignalDF_Base],
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
            p0_df.query(f"param=='amp'")
            .copy(deep=True)
            .loc[:, ["window_idx", "peak_idx", "param", "p0"]]
            .assign(lb=lambda df: df["p0"] * 0.1)
            .assign(ub=lambda df: df["p0"] * 10)
            .drop(["p0"], axis=1)
        )
        # location

        loc = self.get_loc_bounds(
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

        return bounds

    def get_loc_bounds(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
    ):
        """
        Define the upper and lower bounds of the time domain of each peak as the extremes of the window each peak is assigned to. This is achieved by as series of joins and pivots to first label each peak then summarise the range of each window and combine the two. Returns two dataframes containing the labelled upper and lower bound series
        'window_max' and'window_min', respectively.
        """
        
        # get the window each peak belongs to
        peak_df_window_df = (
            peak_df
            .set_index("time_idx")
            .loc[:, ["peak_idx"]] 
            .join([
                window_df
                  .set_index("time_idx")
                  .loc[:,"window_idx"],
                #   signal_df
                #   .set_index("time_idx")
                #   .loc[:,"time"]
                  ])
            .reset_index()
        )
        
        # get time index bounds of each window

        pivot_window_df = window_df.pivot_table(columns="window_idx", values="time_idx", aggfunc=["min", "max"])
        pivot_window_df = pivot_window_df.set_axis(pivot_window_df.columns.set_names("agg", level=0), axis=1).reorder_levels(['window_idx','agg'], axis=1).sort_index(axis=1)

        melt_window_df = pivot_window_df.melt(value_name="time_idx")
        
        # join with signal_df to get time values
        join_df = melt_window_df.set_index('time_idx').join(signal_df.set_index('time_idx').loc[:,'time'], how='left').reset_index()
        
        # join with peak_df_window_df to assign the bound to each peak
        
        interm_tbl = join_df.set_index('window_idx').join(peak_df_window_df.set_index('window_idx').loc[:,['peak_idx']], how='left').reset_index().sort_values(['window_idx','peak_idx'])
    
        # pivot such that columns become 'lb','ub'
        loc_bounds = interm_tbl.pivot_table(values="time", columns="agg", index=["window_idx","peak_idx"]).reset_index().rename({"min": "lb", "max": "ub"}, axis=1, errors="raise")

        # add 'param' label col
        loc_bounds['param']='loc'
        
        # reorder the columns to hint at the index cols
        loc_bounds=loc_bounds.reindex(["window_idx","peak_idx","param","lb", "ub"],axis=1)
        
        # set dtypes
        loc_bounds = loc_bounds.astype({
            "window_idx":pd.Int64Dtype(),
            "peak_idx": pd.Int64Dtype(),
            "param": pd.StringDtype(),
            "lb": pd.Float64Dtype(),
            "ub": pd.Float64Dtype(),
        })
        
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
                window_df.set_index('time_idx').loc[:,'window_idx'],
                how="left",
            )
            .dropna()
            .astype({"window_idx": pd.Int64Dtype()})
        )
        return windowed_signal_df

    @pa.check_types
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
            # na_param_df = param_df.loc[param_df.isna()]
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

    @pa.check_types
    def deconvolve_peaks(
        self,
        signal_df: pt.DataFrame[OutSignalDF_Base],
        peak_df: pt.DataFrame[OutPeakDF_Base],
        window_df: pt.DataFrame[OutWindowDF_Base],
        timestep: np.float64,
    ) -> tuple[pt.DataFrame[OutPoptDF_Base], pt.DataFrame[OutReconDFBase]]:
        windowed_signal_df = self.dataprepper._window_signal_df(signal_df, window_df)

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

        self._param_df = param_df

        popt_df = self._popt_factory(windowed_signal_df, param_df)

        reconstructed_signals = self._reconstruct_peak_signal(
            windowed_signal_df["time"], popt_df
        )

        return popt_df.pipe(pt.DataFrame[OutPoptDF_Base]), reconstructed_signals.pipe(pt.DataFrame[OutReconDFBase])  # type: ignore

    def compile_peak_report(
        self,
        popt_df: pt.DataFrame[OutPoptDF_Base],
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
        windowed_signal_df: pt.DataFrame[OutWindowedSignalBase],
        amp_col: str,
        param_df: pt.DataFrame[OutParamsBase],
    ) -> tuple[
        Annotated[pt.Series[float], "x"],
        Annotated[pt.Series[float], "y"],
        Annotated[pt.Series[float], "p0"],
        Annotated[pt.Series[float], "lb"],
        Annotated[pt.Series[float], "ub"],
    ]:
        """ """
        x = windowed_signal_df.query("window_idx==@window").loc[:, "time"]

        y = windowed_signal_df.query("window_idx==@window").loc[:, amp_col]

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
    ) -> pt.DataFrame[OutPoptDF_Base]:
        popt_list = []

        windows = windowed_signal_df["window_idx"].unique()

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

        return popt_df.pipe(pt.DataFrame[OutPoptDF_Base])

    @pa.check_types
    def _reconstruct_peak_signal(
        self,
        time: npt.NDArray[np.float64],
        popt_df: pt.DataFrame[OutPoptDF_Base],
    )->pt.DataFrame[OutReconDFBase]:
        
        def reconstruct_peak_signal(
            popt_df: pt.DataFrame[OutPoptDF_Base],
            time: pt.Series,
        ):
            params = popt_df.loc[:, ["amp", "loc", "whh", "skew"]].values.flatten()

            try:
                unmixed_signal = self._compute_skewnorm(time.values, *params)

                unmixed_signal_df = pd.merge(
                    popt_df.loc[:, ["peak_idx"]], time, how="cross"
                )

                unmixed_signal_df = unmixed_signal_df.assign(unmixed_amp=unmixed_signal)

            except Exception as e:
                raise RuntimeError(f"{e}\n" f"{params}\n" f"{popt_df.index}\n")
            
            return unmixed_signal_df
                
                
        # remove window_idx from identifier to avoid multiindexed column

        unmixed_df = popt_df.groupby(
            by=["peak_idx"],
            group_keys=False,
        ).apply(
            reconstruct_peak_signal, time
        )  # type: ignore
        
        
        
        return unmixed_df.pipe(pt.DataFrame[OutReconDFBase])

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
