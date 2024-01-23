AMPRAW='amp_raw'
AMPCORR='amp_corrected'

# p0 param column category values
P0AMP='amp'
P0TIME='time'
P0WIDTH='whh_half'
P0SKEW='skew'

# hvplot defaults

from holoviews import opts
import holoviews as hv

hv.extension('bokeh')
opts.defaults(
    opts.Curve(
        height=800,
        width=1200,
        # theme='dark_minimal'
    )
)
