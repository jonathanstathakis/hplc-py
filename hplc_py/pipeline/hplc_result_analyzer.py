"""
A class to **again** extract and format the intermediate and final results of the `cremerlab.hplc`

TODO:
- [ ] format params
"""

from typing import Literal, Any
import polars as pl
from hplc.quant import Chromatogram
import pandas as pd

from hplc_py.common import caching
from cachier import cachier


class HPLCResultsAnalzyer:
    """
    A class that internally runs the Cremerlab hplc-py pipeline to extract the intermediate and final results. Cleans and tabulates the values for easy observation and comparision with my data.
    
    All the action happens in `__init__`, which is seperated into several sections seperated into subclass namespaces (except for definitions):\
    - definitions: table key constants used throughout the tabulations
    - raw: the 'dirty' values extracted straight from the Cremerlab Chromatogram object.
    - tbls: cleaned polars DataFrames ready for observation. Typically normalized where feasible (?) #TODO: actually do this, not true atm
    - views: polars DataFrames useful for comparisons, for example observing the skewnorm parameters 'locaiton','amplitude','scale','skew' (as defined by Cremerlab).
    
    To simplify things, all tables and thus all views time values are expressed in time units as per Cremerlab. There are transformation functions 'to_time' and "to_index" which transform the time to to time index and vice versa, if necessary. (Of couse depending on the irregularity of the time values, this may not always be a correct transformation.)
    
    TODO: documentation
    TODO: define a master key mapping and rename all the tables with that mapping in order to produce unified keys.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cols: dict[str, str] = {"time": "time", "signal": "signal"},
    ):
        """
        does the following:
        - calls fit_chm to generate all of the Cremerlab values
        - defines instance level key name strings for tables
        - defines subclass namespaces for different objects: `raw` for the Cremerlab values, 'tbls' for the cleaned tables #TODO: add views namespace for views.
        """
        self._chm = self.fit_chm(data=data, cols=cols)

        # definitions
        self._P_IDX_ABS = "peak_idx_abs"
        self._PEAK_PARAM_OBSERVED = "peak_param_observed"

        from dataclasses import dataclass

        @dataclass
        class Raw:
            df: pd.DataFrame
            window_props: Any
            int_col: Any
            time_col: Any
            normint: Any
            ranges: Any
            scores: Any
            unmixed_chromatograms: Any
            window_df: Any
            timestep: Any

        # raw

        self._raw = Raw(
            df=self._chm.df,
            window_props=self._chm.window_props,
            int_col=self._chm.int_col,
            time_col=self._chm.time_col,
            normint=self._chm.normint,
            ranges=self._chm.ranges,
            scores=self._chm.scores,
            unmixed_chromatograms=self._chm.unmixed_chromatograms,
            window_df=self._chm.window_df,
            timestep=self._chm._dt,
        )

        # tbls

        @dataclass
        class Tbls:
            peak_map: Any
            curve_fit_inputs: Any
            window_props: Any
            peak_map: Any
            popt: Any

        self.tbls = Tbls(
            curve_fit_inputs=self._clean_params(params=self._chm.params_jono),
            window_props=self._tabulate_window_props(
                window_props=self._chm.window_props
            ),
            peak_map=self._clean_peak_map(peak_map=self._chm._peak_map_jono),
            popt=self._tablulate_popt(peaks=self._chm.peaks),
        )

        # views
        self.view_peak_loc_whh = self._create_view_peak_loc_whh(
            peak_map=self.tbls.peak_map
        )

        self.view_skewnorm_parameter_tbl = self._create_view_skewnorm_parameter_tbl(
            curve_fit_input_params=self.tbls.curve_fit_inputs,
            popt=self.tbls.popt,
            peak_loc_whh=self.view_peak_loc_whh,
        )

        # other

        self.description_dict = {
            self._PEAK_PARAM_OBSERVED: "the peak parameters measured through `scipy.signal`, with an empty 'skew' field as this is not measured but is inferred later",
        }

    def _tform_to_time(
        self,
        df: pl.DataFrame,
        time_idx_key: str,
        new_key: str = "",
    ):
        """
        Convert time index to time units, multiplying by the Cremerlab timestep. new name defaults to `time_idx_key`
        """

        if not new_key:
            new_key = time_idx_key
        out = df.with_columns(pl.col(time_idx_key).mul(self._raw.timestep)).rename(
            {time_idx_key: new_key}
        )

        return out

    def _get_param_synonym_dict(
        self,
        left: Literal["peak_map", "params", "popt"],
        right: Literal["peak_map", "params", "popt"],
    ) -> dict:
        """
        Return a dict mapping the parameter keys between the different definitions of the same parameter.

        :param left: the left side of the mapping.
        :param right: the right side of the mapping

        Each column is the 'param' key each measurementfor that table. param column is a table-agnostic label of that parameter. Pairs of columns can then be converted to dicts where the first column is the key and the second value for submission to replace, rename, etc. see [stack overflow](https://stackoverflow.com/a/75994185)

        Usage:
        ```
        iterator = parameter_synonyms.select('peak_map','params').iter_rows()
        syn_dict = dict(iterator)

        ..
        .rename(syn_dict)
        ..
        ```
        """
        parameter_synonyms = pl.DataFrame(
            data={
                "param": ["maxima", "location", "width", "skew"],
                "peak_map": ["amp", "time", "whh_width", "skew"],
                "params": ["amplitude", "location", "scale", "skew"],
                "popt": ["amplitude", "retention_time", "scale", "skew"],
            }
        )

        synonym_dict = dict(parameter_synonyms.select(left, right).iter_rows())

        return synonym_dict

    def _clean_params(
        self,
        params: pd.DataFrame,
    ) -> pl.DataFrame:
        """
        adds an absolute peak index column from a ranking of w_idx and p_idx as a struct.
        """

        tbl_params = (
            params
            .pipe(pl.from_pandas)
            .with_columns(
            pl.struct(["w_idx", "p_idx"])
            .rank("dense")
            .sub(1)
            .cast(str)
            .alias(self._P_IDX_ABS)
            )
            )  # fmt: skip

        return tbl_params

    def _clean_peak_map(self, peak_map: pd.DataFrame) -> pl.DataFrame:
        """
        Renames "p_idx" to `P_IDX_ABS` value and casts it to str, transforms all the x-axis measurements to time units rather than index units.

        TODO: write a better transform interface to accept lists of column names
        """
        tbl_peak_map = (
            peak_map.pipe(pl.from_pandas)
            .rename({"p_idx": self._P_IDX_ABS})
            .with_columns(pl.col(self._P_IDX_ABS).cast(str))
            .pipe(self._tform_to_time, "time_idx", "time")
            .pipe(self._tform_to_time, "whh_width")
            .pipe(self._tform_to_time, "whh_left")
            .pipe(self._tform_to_time, "whh_right")
            .pipe(self._tform_to_time, "pb_width")
            .pipe(self._tform_to_time, "pb_left")
            .pipe(self._tform_to_time, "pb_right")
            .melt(id_vars=[self._P_IDX_ABS], value_name="value", variable_name="param")
        )

        return tbl_peak_map

    def _create_view_peak_loc_whh(self, peak_map: pl.DataFrame) -> pl.DataFrame:
        """
        starting with `peak_map`, filters for 'time', 'amp' and 'whh_width', pivots on P_IDX_ABS, adds an empty 'skew' field, then melts with P_IDX_ABS as index, naming the values as "param_peak_observed". The naming is intended to differentiate this value column from popt and params if they are joined.
        """

        peak_loc_whh = pl.DataFrame()

        peak_loc_whh = (
            peak_map.filter(pl.col("param").is_in(["time", "amp", "whh_width"]))
            .pivot(index=self._P_IDX_ABS, columns="param", values="value")
            .with_columns(pl.lit(None).alias("skew"))
            .melt(
                id_vars=self._P_IDX_ABS,
                value_name="peak_param_observed",
                variable_name="param",
            )
        )

        return peak_loc_whh

    def _tablulate_popt(self, peaks: pd.DataFrame | None) -> pl.DataFrame:
        """
        Starting with Cremerlab `peaks` columns "retention_time", "scale","skew","amplitude","peak_id" (excluding "area"), renames "peak_id" to "p_idx", "retention_time" to "loc", subtracts 1 from "p_idx" so that it starts at zero, casts to string and renames it `_P_IDX_ABS` value, then renames the parameter column keys to match the arams table, finally melting with `_P_IDX_ABS` as index, "param" as variable name, "popt" as value name to differentiate it from peak map or input params values.
        """
        if not isinstance(peaks, pd.DataFrame):
            raise ValueError("peaks is empty")

        tbl_popt = (
            peaks.pipe(pl.from_pandas)
            .select(
                [
                    pl.col("peak_id").sub(1).cast(str).alias(self._P_IDX_ABS),
                    pl.col("retention_time"),
                    pl.col("scale"),
                    pl.col("skew"),
                    pl.col("amplitude"),
                ]
            )
            .rename(self._get_param_synonym_dict("popt", "params"))
            .melt(id_vars=self._P_IDX_ABS, variable_name="param", value_name="popt")
        )

        return tbl_popt

    @cachier(hash_func=caching.custom_param_hasher, cache_dir=caching.CACHE_PATH)
    def fit_chm(
        self, data: pd.DataFrame, cols: dict[str, str] = {"time": "x", "signal": "y"}
    ):
        """
        Initializes a Cremerlab Chromatogram object and calls its `fit_peaks`, and `assess_fit` methods to generate the intermediate and final results, returns the Chromatogram object in the 'asses_fit' state.
        """

        chm = Chromatogram(file=data, cols=cols)
        chm.fit_peaks()
        chm.assess_fit()

        return chm

    def _create_view_skewnorm_parameter_tbl(
        self,
        curve_fit_input_params: pl.DataFrame,
        popt: pl.DataFrame,
        peak_loc_whh: pl.DataFrame,
    ):
        """
        Combine the curve fit parameter inputs with the popt in one table.

        in this view, the whh_width is dubbed scale as well, even though its not scale, its the width at half height
        """

        peak_loc_whh_ = peak_loc_whh.with_columns(
            pl.col("param").replace(self._get_param_synonym_dict("peak_map", "params"))
        )

        view_skewnorm_parameters = (
            peak_loc_whh_.join(curve_fit_input_params, on=[self._P_IDX_ABS, "param"])
            .join(
                popt.select([self._P_IDX_ABS, "param", "popt"]),
                on=[self._P_IDX_ABS, "param"],
            )
            .select(
                self._P_IDX_ABS,
                "w_idx",
                "p_idx",
                "param",
                "peak_param_observed",
                "lb",
                "p0",
                "ub",
                "popt",
            )
            .sort([self._P_IDX_ABS, "param"])
        )

        return view_skewnorm_parameters

    def _tabulate_window_props(self, window_props: dict | None) -> pl.DataFrame:
        """
        unpack the window_props dict into a dataframe

        Window props include the time and signal, the number of peaks in the window, and the skewnorm parameters of each contained peak. 3 different types of data. Considering window_df contains the windowing as well, discard the signals and just get the peak props.
        """
        dfs = []
        for k, v in window_props.items():  # type: ignore

            new_dict = {
                kk: vv
                for kk, vv in v.items()
                if kk not in ["time_range", "signal", "num_peaks", "signal_area"]
            }

            df = pl.DataFrame(new_dict).with_columns(pl.lit(k).alias("w_idx"))

            dfs.append(df)

        tbl_windowed_peak_params = (
            pl.concat(dfs)
            .with_row_index(name="p_idx")
            .select(
                pl.col("w_idx").sub(1),
                pl.col("p_idx").cast(int),
                pl.col(["amplitude", "location", "width"]),
            )
            .melt(
                id_vars=["w_idx", "p_idx"],
                variable_name="param",
                value_name="popt_input",
            )
        )

        return tbl_windowed_peak_params
