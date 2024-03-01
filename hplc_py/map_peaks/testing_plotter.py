import holoviews as hv
import hvplot
import numpy as np
import pandas as pd
import dataclass
from typing import Any
from dataclasses import asdict

x1 = np.arange(0, 10, 1)
y1 = np.random.rand(len(x1))

df = pd.DataFrame(dict(x=x1, y=y1))

import hvplot.pandas  # noqa

scatter = df.hvplot(x="x", y="y", kind="scatter", label="scatter")
line = df.hvplot(x="x", y="y", kind="line", label="line")


@dataclass
class Plotter:
    scatter: Any

    line: Any


Plotter
plotter = Plotter(scatter=scatter, line=line)
plotter
overlay = nd.Overlay(list(asdict(plotter).values()))
overlay = hv.Overlay(list(asdict(plotter).values()))
overlay
hvplot.show(overlay)
