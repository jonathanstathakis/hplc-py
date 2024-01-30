from holoviews import opts
import holoviews as hv

import polars as pl
import pandas as pd

pd.options.display.max_columns = 50

pl.Config(
    set_tbl_rows=50,
    set_tbl_cols=20,
    )

AMP='amp'
AMPRAW='amp_raw'
AMPCORR='amp_corrected'

# p0 param column category values
P0AMP='amp'
P0TIME='time'
P0WIDTH='whh_half'
P0SKEW='skew'

# hvplot defaults


hv.extension('bokeh') #type: ignore

opts.defaults(
    opts.Curve(
        height=800,
        width=1200,
        # theme='dark_minimal'
    )
)
