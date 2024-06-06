import hvplot
import polars as pl
import pandas as pd
from hplc_py.common import definitions as com_defs
from hplc_py.baseline_correction import definitions as bc_defs


class VizBCorr:
    """
    simple class designed to be inherited by BaselineCorrection
    """

    def viz_baseline_correction(
        self,
        signals: pd.DataFrame,
    ):
        """
        Expected to be called after `correct_baseline` to plot the internally stored signals frame, returning a hvplot plot_obj which can be drawn with `hvplot.show`.
        """
        
        plot_obj = signals.pipe(pl.from_pandas).plot(
            x="idx",
            y="amp",
            by=bc_defs.KEY_SIGNAL,
            grid=True,
            legend="top",
            title="Signal Baseline Correction",
            height=750, width=1500,
        )

        return plot_obj
