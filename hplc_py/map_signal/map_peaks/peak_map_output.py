import polars as pl
from hplc_py.map_signal.map_peaks import viz_hv
from hplc_py.common import common_schemas as com_schs
from hplc_py.map_signal.map_peaks import schemas as mp_schs
from hplc_py import precision
from hplc_py.map_signal.map_peaks.contour_line_bounds import ContourLineBounds
from pandera.typing.polars import DataFrame as Polars_DataFrame
from hplc_py import transformers
from hplc_py.common import common

class PeakMap(precision.Precision):

    def __init__(
        self,
        maxima: Polars_DataFrame[mp_schs.Maxima],
        contour_line_bounds: ContourLineBounds,
        widths: Polars_DataFrame[mp_schs.Widths],
        X,
    ):
        self._maxima = maxima
        self._contour_line_bounds = contour_line_bounds
        self._widths = widths
        self._X = X

    @property
    def maxima(self):
        return self._maxima

    @maxima.getter
    def maxima(self):
        return self._maxima

    def _set_maxima(self, new_maxima):
        self._maxima = new_maxima

    @property
    def contour_line_bounds(self):
        return self._contour_line_bounds

    def _set_contour_line_bounds(self, new_contour_line_bounds):
        self._contour_line_bounds = new_contour_line_bounds

    @property
    def widths(self):
        return self._widths

    def _set_widths(self, new_widths):
        self.widths = new_widths

    @widths.getter
    def widths(self):
        return self._widths.with_columns(pl.col("value").round(self._precision))

    @property
    def X(self):
        return self._X

    @X.getter
    def X(self):
        return self._X.with_columns(pl.col("X").round(self._precision))

    def as_viz(self):
        return viz_hv.PeakMapViz(self)

    def __repr__(self):
        return f"contains the following tables:\n\tmaxima:\n\t\tcolumns: {self.maxima.columns}\n\t\tshape: {self.maxima.shape}\n\tcontour_line_bounds\n\t\tcolumns: {self.contour_line_bounds.bounds.columns}\n\t\tshape: {self.contour_line_bounds.bounds.shape}\n\twidths:\n\t\tcolumns: {self.widths.columns}\n\t\tshape: {self.widths.shape}\n"


class WindowedPeakMap(common.PolarsDataObjMixin):

    def __init__(self, maxima, contour_line_bounds, widths):

        self.maxima = maxima
        self.contour_line_bounds = contour_line_bounds
        self.widths = widths

    def as_viz(self):
        return viz_hv.PeakMapViz(self)
    
    def __repr__(self):
        tbl_names = ['maxima', 'contour_line_bounds', 'widths']
        attrs = ['columns','shape']
        
        repr_str = self.tbl_repr_formatter(tbl_names=tbl_names, tbl_props=attrs)
        
        return repr_str
