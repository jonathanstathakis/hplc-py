
import pandera as pa
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import pytest
from matplotlib.figure import Figure
from pandera.typing import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import PSignals, SignalDFLoaded, RSignal
from hplc_py.show import Show, PlotSignal

from matplotlib.axes import Axes as Axes

pl.Config(set_tbl_cols=50)


class TestShow:
    """
    Test the Show class methods
    """

    @pytest.fixture
    def fig_ax(self):
        return plt.subplots(1)

    @pytest.fixture
    def fig(self, fig_ax):
        return fig_ax[0]
    
    @pytest.fixture
    def ax(self, fig_ax):
        return fig_ax[1]
    
    def test_plot_raw_signal(
        self,
        ax: Axes,
        in_signal: DataFrame[SignalDFLoaded],
        time_col,
        amp_col,
    ):
        PlotSignal(df=in_signal, x_colname=time_col, y_colname=amp_col, label=amp_col, ax=ax)._plot_signal_factory(
            
        )
        
        plt.show()

    def test_plot_recon_signal(
        self,
        ax: Axes,
        r_signal: DataFrame[RSignal],
        time_col,
        amp_col,
    ):
        
        breakpoint()
        
        PlotSignal(df=r_signal, x_colname=time_col, y_colname=amp_col, label='recon',ax=ax)._plot_signal_factory(
            
        )
        
        plt.show()
        
    @pa.check_types
    def test_plot_individual_peaks(
        self,
        show: Show,
        ax: Axes,
        psignals: DataFrame[PSignals],
    ):

        pass
        # plt.show()

    # @pytest.mark.skip(reason="needs to be updated to match current implementation")
    # def test_plot_overlay(
    #     self,
    #     show: Show,
    #     fig_ax: tuple[Figure, Any],
    #     signal_df: DataFrame,
    #     psignals: DataFrame[PSignals],
    # ):
    #     fig = fig_ax[0]
    #     ax = fig_ax[1]
    #     show.(
    #         signal_df,
    #         ax,
    #     )
    #     show.plot_reconstructed_signal(
    #         psignals,
    #         ax,
    #     )
    #     show.plot_individual_peaks(
    #         psignals,
    #         ax,
    #     )
