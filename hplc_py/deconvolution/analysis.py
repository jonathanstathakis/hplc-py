"""
Results analysis
"""

import hvplot
import hvplot.pandas
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
        data: pl.DataFrame,
        recon,
    ):
        data = (
            data.pipe(check_is_polars)
            .rename({"amplitude": "mixed"})
            .join(recon, on=self._x_unit)
            .melt(
                id_vars=["w_type", "w_idx", self._x_unit],
                variable_name="signal",
                value_name="amplitude",
            )
        )
        return data


def check_is_polars(df):
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected polars dataframe, got {type(df)}")
    return df


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
        reporter = fit_assessment.Reporter(data=data, key_time=self._x_unit)
        fit_report = reporter.calc_fit_report()
        
        return fit_report

from hplc_py.map_peaks import map_peaks, viz as mp_viz

class InspectorViz:
    def __init__(self, X, peak_map):
        self.peak_map = mp_viz.VizPeakMapFactory(
            X=X, peak_map=peak_map,
        )

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
        peak_map: DataFrame,
        x_unit: str,
    ):

        self._keys_tbl_mixed_signal: KeysTblMixedSignal = dc_defs.keys_tbl_mixed_signal

        self.reconstructor = Reconstructor(X_w=signal, popt=popt, x_unit=x_unit)
        tbl_signal_unmixed = self.reconstructor.unmixed_signals

        analyzer_in = self.reconstructor.mixed_signals.pipe(
            deconvolution.get_active_signal_as_mixed,
            x_unit=x_unit,
            active_signal="mixed",
            keys=dc_defs.keys_tbl_mixed_signal,
        )

        self.analyzer = Analyzer(data=analyzer_in, x_unit=x_unit)
    
    

    def plot_results(self):
        self.mixed_signals_overlay = self._plot_mixed_signals_overlay(
            mixed_signals=self.reconstructor.mixed_signals
        )

        self.unmixed_signals_overlay = self._plot_unmixed_signals_overlay(
            unmixed_signals=self.reconstructor.unmixed_signals
        )

        hvplot.show(self.mixed_signals_overlay * self.unmixed_signals_overlay)

    def _plot_unmixed_signals_overlay(self, unmixed_signals):

        hvplot.extension(backend="bokeh")

        unmixed_signals_area_overlay = (
            unmixed_signals.pipe(pl.from_pandas)
            # .pivot(index="x", columns="p_idx", values="unmixed")
            # .pipe(lambda df: df if breakpoint() else df)
            .to_pandas().hvplot.area(
                x="time",
                y="unmixed",
                by="p_idx",
                height=1000,
                width=2000,
                alpha=0.5,
                stacked=False,
            )
        )
        return unmixed_signals_area_overlay

    def _plot_mixed_signals_overlay(self, mixed_signals):

        mixed_signals_overlay = mixed_signals.plot(x="time", y="amplitude", by="signal")

        return mixed_signals_overlay