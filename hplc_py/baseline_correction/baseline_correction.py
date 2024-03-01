import polars as pl
import pandas as pd
from hplc_py.baseline_correction.compute_background import ComputeBackground
from hplc_py.baseline_correction import definitions as bc_defs
from hplc_py.baseline_correction.viz import VizBCorr
from hplc_py.common import definitions as com_defs


class BaselineCorrection(ComputeBackground, VizBCorr):
    """
    A 'batteries included' baseline correction module which takes a 1D signal and computes the background, subtracting it and returning a dataframe of time index, raw signal, background, and corrected signal.

    It includes a viz method via hvplot which will produce a plot object that displays an overlay of the raw, corrected and background curves. To use, call `hvplot.show(plot_obj)`
    """

    def __init__(
        self,
        n_iter: int,
        window_size: float,
        verbose: bool,
    ):
        super().__init__(n_iter=n_iter, window_size=window_size, verbose=verbose)

    def correct_baseline(
        self,
    ) -> pd.DataFrame:
        """
        call after `fit` to perform the transform but also assemble the output data alongside the input data in a frame ready for returning and for viz internally
        TODO: schema the output
        """
        if self._X.size == 0:
            raise AttributeError(
                "internal X array is empty, probably need to call `fit` first"
            )

        self.transform()

        self.signals = (
            pl.DataFrame(
                {
                    bc_defs.KEY_RAW: self._X,
                    bc_defs.KEY_BACKGROUND: self.background,
                    bc_defs.KEY_CORRECTED: (self._X - self.background),
                }
            )
            .with_row_index(name=bc_defs.X_IDX)
            .with_columns(pl.col(bc_defs.X_IDX).cast(int))
            .melt(
                id_vars=bc_defs.X_IDX,
                variable_name=bc_defs.KEY_SIGNAL,
                value_name=bc_defs.X,
            )
            .to_pandas()
        )

        return self.signals
