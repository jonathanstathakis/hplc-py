"""
Results analysis
"""

import pandera as pa
from pandera.typing import DataFrame
from hplc_py.deconvolution import fit_assessment
from hplc_py.deconvolution import deconvolution as decon
import polars as pl
import pandas as pd

from hplc_py.deconvolution import schemas as dc_schs, definitions as dc_defs
from hplc_py.deconvolution import deconvolution


from hplc_py.deconvolution.definitions import KeysTblMixedSignal


class Reconstructor:
    """
    A simple class containing the reconstruction actions.

    After initialization, call `get_mixed_signals` to get a dataset containing the input data and the reconstructed signal from the unmixed peak signals. Call `get_unmixed_signals` to get the unmixed peak signals in long form.
    """

    @pa.check_types(lazy=True)
    def __init__(
        self,
        X_w: DataFrame[dc_schs.ReconstructorSignalIn],
        popt: DataFrame[dc_schs.ReconstructorPoptIn],
        x_unit,
    ):

        self._X_w = X_w.pipe(pl.from_pandas)
        self._x_unit = x_unit

        self.unmixed_signals = deconvolution.construct_unmixed_signals(
            X_w=X_w,
            popt=popt,
            x_unit=x_unit,
        )

        recon = deconvolution.reconstruct_signal(
            peak_signals=self.unmixed_signals, x_unit=x_unit
        )

        self.mixed_signals = self._X_w.pipe(
            self._add_recon_to_data, recon.pipe(pl.from_pandas)
        )

    def _add_recon_to_data(
        self,
        data,
        recon,
    ):
        data = (
            data
            .rename({"amplitude": "mixed"})
            .join(recon, on="x")
            .melt(
                id_vars=["w_type", "w_idx", "x"],
                variable_name="signal",
                value_name="amplitude",
            )
        )
        return data


class Analyzer:

    @pa.check_types(lazy=True)
    def __init__(
        self,
        data: DataFrame[dc_schs.ActiveSignal],
        x_unit: str,
    ):
        """
        :param data: a polars dataframe containing the columns relevant to `calc_fit_report`.
        """
        self._keys_tbl_mixed_signal: KeysTblMixedSignal = dc_defs.keys_tbl_mixed_signal

        self._x_unit = x_unit

        self.fit_report = data.pipe(self._retrieve_fit_report).pipe(pl.from_pandas)

    def get_fit_report(self):
        return self.fit_report

    @pa.check_types
    def _retrieve_fit_report(
        self,
        data: DataFrame[dc_schs.ActiveSignal],
    ):

        fit_report = fit_assessment.calc_fit_report(
            data=data,
        )

        return fit_report


class Inspector:
    """
    A development tool for inspecting intermediate popt values.

    Actions:
    - accept a x series and popt table
    - construct the unmixed signals
    - reconstuct a mixed signal
    - generate fit scores
    """

    @pa.check_types(lazy=True)
    def __init__(
        self,
        signal: DataFrame[dc_schs.ReconstructorSignalIn],
        popt: DataFrame[dc_schs.ReconstructorPoptIn],
        x_unit: str,
    ):

        self._keys_tbl_mixed_signal: KeysTblMixedSignal = dc_defs.keys_tbl_mixed_signal

        reconstructor = Reconstructor(X_w=signal, popt=popt, x_unit=x_unit)

        tbl_signal_unmixed = reconstructor.unmixed_signals

        analyzer_in = reconstructor.mixed_signals.pipe(
            deconvolution.get_active_signal_as_mixed,
            x_unit=x_unit,
            active_signal="mixed",
            keys=dc_defs.keys_tbl_mixed_signal,
        )

        self.analyzer = Analyzer(
            data=analyzer_in, x_unit=x_unit
        )
        
