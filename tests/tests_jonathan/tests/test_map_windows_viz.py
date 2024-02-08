import matplotlib.pyplot as plt
import pytest
from pandera.typing.pandas import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import (
    WindowedSignal,
)
from hplc_py.show import DrawPeakWindows, SignalPlotter


@pytest.fixture
def dpw() -> DrawPeakWindows:
    dpw = DrawPeakWindows()
    return dpw

@pytest.fixture
def sp() -> SignalPlotter:
    sp = SignalPlotter()
    return sp

def test_map_windows_plot(
    dpw: DrawPeakWindows,
    sp: SignalPlotter,
    ws_: DataFrame[WindowedSignal],
) -> None:
    fig, ax = plt.subplots()

    sp.plot_signal(
        ax,
        ws_,
        str(WindowedSignal.time),
        str(WindowedSignal.amp_corrected),
        "signal",
    )

    dpw.draw_peak_windows(
        ws_,
        ax,
    )

    plt.show()
