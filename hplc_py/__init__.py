from holoviews import opts
import holoviews as hv

import polars as pl
import pandas as pd

import os

from traitlets import default

## Defaults

### Project

ROOT = os.path.dirname(os.path.abspath(__file__))
SCHEMA_CACHE = os.path.join(ROOT, "schema_cache")

### Pandas

pd.options.display.max_columns = 50

### Polars

pl.Config(
    set_tbl_rows=20,
    set_tbl_cols=20,
)

### hvplot

#### Extension

hv.extension("bokeh")  # type: ignore

#### Theme

# See [here](https://notebook.community/redlegjed/cheatsheets/holoviews_cheatsheet)
# cant find official docs on the matter
from bokeh.themes import built_in_themes

hv.renderer("bokeh").theme = built_in_themes["dark_minimal"]

#### Plots

# again, cant find anything in official docs. see [here](https://stackoverflow.com/a/57616004)

import holoviews as hv

# the hunt continues. [this forum post](https://discourse.holoviz.org/t/set-default-plot-size-for-all-plot-types/5722/7) suggests that the following can be used to access global settings, but it doesnt work

default_opts = dict(width=1200, height=800, responsive=True)
hv.opts.defaults(opts.Curve(**default_opts), opts.Area(**default_opts))

# Pandera
