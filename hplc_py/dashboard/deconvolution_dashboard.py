"""
2024-02-21 12:10:42

# Building a dashboard

The dashboard should have:
- [x] the original plot
- [x] the baseline corrected plot
- [x] the background
- [ ] the peaks
- [ ] the windows
- [ ] the reconstruction
- [ ] the peak signal report
- [ ] the fit score report

To do that we need to:
- [ ] get the data
- [ ] baseline correct the data
- [ ] plot the baseline correction
- [ ] map the peaks
- [ ] display the peak map table
- [ ] plot the peak map
- [ ] map the windows
- [ ] display the window bounds table
- [ ] plot the windowing
- [ ] deconvolve
- [ ] plot the deconvolved peaks
- [ ] display the peak report
- [ ] display the fit score report.

As we are displaying a process, it will be useful to display as the process is occuring - for example, once the signal is baesline corrected, display, once the peaks are mapped, display, once the windows are mapped, display, etc. Of course all but the deconvolution stages are basically instantaneous, but its a good design philosophy. Unfortunately I have currently vertically stacked the component modules inside the deconvolution module. To avoid modifying source code while developing the viz, I will viz each submodule seperately, then run them again within the deconvolution call, then factor that out. The ideal long term goal would be to rewrite the deconvolution module to be used within a groupby operation window to window - i.e. it will not require information about the windowing, and deconvolve each window seperately. This way i can factor out the windowing if it is deemed superfluous. But that is downstream. Lets build these up one by one, starting with the raw plot.


"""

from typing import Any
import panel as pn
import hvplot
import polars as pl
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

import tests.tests_jonathan.tests.test_baseline_correction.conftest
from hplc_py.viz import plot_signal
from hplc_py.dashboard.dsets import get_asschrom_dset
from hplc_py.baseline_correction.baseline_correction import BaselineCorrection
from hplc_py.baseline_correction.baseline_correction import compute_n_iter
from hplc_py.common import compute_timestep
from hplc_py.map_peaks.map_peaks import PeakMapper
from pandera.typing import DataFrame
from hplc_py.common.common_schemas import X_Schema
from hplc_py.map_peaks.schemas import PeakMap

pn.extension("ipywidgets")


# pn.extension(design="material", theme="dark")
def get_corrected_background(X: DataFrame[X_Schema]) -> pd.DataFrame:
    """
    add the background and corrected X to the dset. returns a 3 column frame of 'raw',
    'corrected', 'background'.
    """

    X = X.pipe(pl.from_pandas).select("X").to_series().to_numpy()

    # n_iter = compute_n_iter(window_size=5, timestep=timestep)
    n_iter = 5

    cb = BaselineCorrection()
    cb.fit(X=X, n_iter=n_iter)
    cb.transform()
    background = tests.tests_jonathan.tests.test_baseline_correction.conftest.background
    corrected = cb.corrected

    out: pd.DataFrame = (
        X.pipe(pl.from_pandas)
        .with_columns(
            pl.Series(name="background", values=background),
            pl.Series(name="corrected", values=corrected),
        )
        .to_pandas()
    )

    return out


def get_peak_map(X: DataFrame[X_Schema]) -> DataFrame[PeakMap]:
    mp = PeakMapper()
    mp.fit(X=X)
    mp.transform()
    peak_map = mp.peak_map

    return peak_map


def get_data(dset: pd.DataFrame) -> dict[str, Any]:
    """
    Assemble all the data required for the dashboard into a dict to pass to a plotting
    function
    """
    data_dict = {}
    # X
    X = prepare_X(dset)

    # X baseline correction information
    X_with_corrected_data = get_corrected_background(X=X)

    # peak map

    peak_map: DataFrame[PeakMap] = get_peak_map(X=X)

    return data_dict


def display_dashboard(dset: pd.DataFrame):
    """
    [launching a server dynamically](https://panel.holoviz.org/how_to/server/programmatic.html)

    [according to this article](https://holoviews.org/user_guide/Deploying_Bokeh_Apps.html) "any .py or .ipynb that attaches a plot to Bokeh's `curdoc` can be deployed using `panel serve`. They do this by for example wrapping the object with `pn.panel` then calling its `pn.panel.servable()`.

    2024-02-21 14:18:55 - matplotlib doesnt play nice with panel, whatever the docs say. Since polars has elected to support hvplot over matplotlib, its probably time to factor out matplotlib. Dont worry about recreating the peak line plots atm, if we want we can gen a static version, but do recreate the windowing.

    The dashboard will have two concurrent processes - data manipulation and display, with data manipulation feeding into display functions as we go.

    """

    plot_raw_signal = dset_pl.plot(x="time", y=["amp"], label="raw_signal")

    plot_peak_map = peak_map.pipe(pl.from_pandas).plot(x="X_idx", y="maxima")

    # plot_corrected_signal = dset_pl.plot(x="time", y=["corrected"], label="corrected")

    col1 = pn.panel(
        [plot_raw_signal, plot_peak_map],
    )
    col1.servable()


def prepare_X(dset):
    X: DataFrame[X_Schema] = (
        dset.pipe(pl.from_pandas)
        .with_row_index(name="X_idx")
        .rename({"corrected": "X"})
        .select(pl.col("X_idx").cast(pl.Int64), "X")
        .to_pandas()
        .pipe(DataFrame[X_Schema])
    )

    return X


def main():
    get_data(dset=get_asschrom_dset())


if __name__.startswith("bokeh"):
    # [syntax source](https://discourse.holoviz.org/t/using-panel-serve-renders-blank-page/1887)
    display_dashboard(dset=get_asschrom_dset())


if __name__ == "__main__":
    main()
