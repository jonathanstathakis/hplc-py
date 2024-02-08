import polars as pl
import pandera as pa
import distinctipy
from typing import Self
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from numpy import float64
from numpy.typing import NDArray
from pandera.typing import DataFrame, Series
from matplotlib.lines import Line2D

from hplc_py.hplc_py_typing.hplc_py_typing import PeakMap, X_Schema
from hplc_py.hplc_py_typing.type_aliases import rgba
from hplc_py.io_validation import IOValid


class PlotCore(IOValid):
    def __init__(self, ax: Axes, df: pd.DataFrame):
        plt.style.use("ggplot")

        self.df = df
        self.ax = ax
        self.colors = distinctipy.get_colors(len(self.df))

    def add_proxy_artist_to_legend(
        self,
        label: str,
        line_2d: Line2D,
    ):
        handles, _ = self.ax.get_legend_handles_labels()
        legend_entry = self._set_legend_entry(label, line_2d)
        new_handles = handles + [legend_entry]
        self.ax.legend(handles=new_handles)

        return self

    def _set_legend_entry(self, label: str, line_2d: Line2D):
        """
        Creates an empty Line2D object to use as the marker for the legend entry.

        Used to generate a representation of a marker category when there are too
        many entries for a generic legend.

        :param label: the legend entry text
        :type label: str
        :param line_2d: a representative line_2d of the plotted data to extract marker
        information from. Ensures that the legend marker matches what was plotted.
        """

        legend_marker = line_2d.get_marker()
        legend_markersize = line_2d.get_markersize()

        # TODO: modify this to a more appropriate method of choosing a 'representative'
        # color
        legend_color = self.colors[-1]
        legend_markeredgewidth = line_2d.get_markeredgewidth()
        legend_markeredgecolor = line_2d.get_markeredgecolor()

        legend_entry = Line2D(
            [],
            [],
            marker=legend_marker,
            markersize=legend_markersize,
            color=legend_color,
            markeredgewidth=legend_markeredgewidth,
            markeredgecolor=legend_markeredgecolor,
            label=label,
            ls="",
        )

        return legend_entry


class PeakMapViz(IOValid):
    """
    Provides a number of chainable plot functions to construct piece-wise a overlay plot of the properties mapped via MapPeaks.

    The ax the overlay is plotted on can be accessed through the `.ax` get method.
    """

    def __init__(
        self,
        df: DataFrame,
        x_colname: str,
        ax: Axes = plt.gca(),
    ):
        self.df = df.copy(deep=True)
        self.x_colname = x_colname
        self.ax = ax
        self.pp = PeakPlotter(ax, df.copy(deep=True))
        self.wp = WidthPlotter(ax)

    def plot_whh(
        self,
        y_colname: str,
        left_colname: str,
        right_colname: str,
        marker: str = "v",
        ax: Axes = plt.gca(),
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Create whh plot specifically.
        """

        label = "whh"

        wp = WidthPlotter(ax)

        wp.plot_widths(
            df=self.df,
            y_colname=y_colname,
            left_x_colname=left_colname,
            right_x_colname=right_colname,
            left_y_key="",
            right_y_key="",
            marker=marker,
            label=label,
            plot_kwargs=plot_kwargs,
        )

        return self

    def plot_bases(
        self,
        y_colname: str,
        left_colname: str,
        right_colname: str,
        ax: Axes,
        marker: str = "v",
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Plot peak bases.
        """

        label = "bases"
        wp = WidthPlotter(ax)

        wp.plot_widths(
            df=self.df,
            y_colname=y_colname,
            left_x_colname=left_colname,
            right_x_colname=right_colname,
            left_y_key="",
            right_y_key="",
            marker=marker,
            label=label,
            plot_kwargs=plot_kwargs,
        )

        return self

    def _plot_peak_factory(
        self,
        x: NDArray[float64],
        y: NDArray[float64],
        label: str,
        color: rgba,
        ax: Axes,
        plot_kwargs: dict[str, Any] = {},
    ) -> Axes:
        """
        plot each peak individually.
        """
        ax.plot(
            x,
            y,
            c=color,
            marker="7",
            linestyle="",
            **plot_kwargs,
            alpha=0.5,
            markeredgecolor="black",
        )

        return ax


class PeakPlotter(PlotCore):
    def __init__(self, ax: Axes, df: pd.DataFrame):
        super().__init__(ax, df)

    def plot_peaks(
        self,
        x_colname: str,
        y_colname: str,
        label: str = "maxima",
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Plot peaks from the peak map, x will refer to the time axis, y to amp.
        """

        self.check_keys_in_index([x_colname, y_colname], self.df.columns)

        for i, (idx, row) in enumerate(self.df.iterrows()):
            self.ax.plot(
                x_colname,
                "maxima",
                data=row,
                marker="o",
                c=self.colors[i],
                markeredgecolor="black",
                label="_",
            )

            self.ax.annotate(
                text=str(idx),
                xy=(row["X_idx"], row["maxima"]),
                ha="center",
                va="top",
                textcoords="offset pixels",
                xytext=(0, 40),
            )

        self.add_proxy_artist_to_legend(label, self.ax.lines[-1])

        return self


import abc

class InterfacePipeline(metaclass=abc.ABCMeta):
    """
    A basic template for pipelines - objects that take one or more inputs and produce
    one output, with internal validation via Pandera DataFrameModel schemas.
    """
    
    @abc.abstractmethod
    def load_pipeline(self, **kwargs):
        """
        load the pipeline with necessary input for `run_pipeline`
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def run_pipeline(self):
        """
        Execute the pipeline for the input of `load_pipeline`
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _set_schemas(self):
        """
        Initialise defined schemas as objects of self.
        """
        raise NotImplementedError

class PeakMapInterface(InterfacePipeline, IOValid):
    """
    Contains methods to organise the peak map data for plotting.
    """

    @pa.check_types
    def __init__(self, peak_map: DataFrame[PeakMap], X: DataFrame[X_Schema]):
        """
        Initialise peak map and X as polars if not already. Up to the user to ensure that nothing lost in transition. Preference is to pass a polars frame rather than pandas.
        """
        if not self._is_polars(peak_map):
            self.peak_map = pl.from_pandas(peak_map)
        else:
            self.peak_map = peak_map
        if not self._is_polars(X):
            self.X = pl.from_pandas(X)
        else:
            self.X = X


class Pipe_Peak_Widths_To_Long(IOValid):
    def __init__(self):
        self._set_schemas()

    @pa.check_types(lazy=True)
    def load_pipeline(
        self,
        peak_map: DataFrame[PeakMap],
        idx_cols: list[str] = ["p_idx"],
        x_cols: list[str] = ["whh_left", "whh_right", "pb_left", "pb_right"],
        y_cols: list[str] = ["whh_height", "pb_height"],
    )->Self:
        
        # internal logic uses polars, but currently pandera does not support them.
        if not self._is_polars(peak_map):
            self._peak_map = pl.from_pandas(peak_map)
        else:    
            self._peak_map = peak_map
        
        self._idx_cols = idx_cols
        self._x_cols = x_cols
        self._y_cols = y_cols
        
        return self
    
    def run_pipeline(
        self,
    )->Self:
        """
        # Wide Peak Map to Long

        Take the wide peak map, the output of MapPeaks, and rearrange into a long format
        with one row per observation, suitable for plotting. Returns a long format frame
        ready for row iteration.

        ## Warning

        This pipeline relies on the x and y columns to share a common prefix, i.e.
        'whh_', or 'pb_'. After melting, the pipeline will split the label column
        into prefix and suffix columns. The prefix columns will be dubbed
        'peak props' and the suffix will be dubbed 'geoprops'. Geoprops represent
        the properties of the width contour line, left, right, height, and 'peak_props'
        represent the inferred peak properties, 'whh', 'pb' (peak bases), etc.

        The internal schemas will enforce this expecation, and the pipeline makes no
        effort to account for deviations from that structure.

        :param peak_map: a wide table containing the peak mapping data. check schema for more info.
        :type peak_map: DataFrame[PeakMap]
        :param idx_cols: the primary key column of the peak_map - likely the peak idx
        :type param_cols: list[str]
        :param x_cols: the columns containing x values - the time locations
        :type x_cols: list[str]
        :param y_cols: the columns containing y values corresponding to x_cols
        :type y_cols: list[str]
        """            
            
        # organise the peak map into left and right frames, where the right frame will be broadcasted so that the
        # left and right ips for each measurement have their corresponding y value.

        x_pm = self._peak_map.select(self._idx_cols + self._x_cols)
        y_pm = self._peak_map.select(self._idx_cols + self._y_cols)

        self.__sch_pm_width_in_x.validate(x_pm.to_pandas(), lazy=True)
        self.__sch_pm_width_in_y.validate(y_pm.to_pandas(), lazy=True)

        # melt both x and y frames

        def melt_peakprop_label(df: pl.DataFrame):
            out = (
                df.melt(id_vars="p_idx", variable_name="peakprop", value_name="msnt")
                .with_columns(pl.col("peakprop").str.split(by="_").list.to_struct())
                .unnest("peakprop")
                .rename({"field_0": "peakprop", "field_1": "geoprop"})
            )

            return out

        x_pm_long = x_pm.pipe(melt_peakprop_label).rename({"msnt": "x"})
        y_pm_long = y_pm.pipe(melt_peakprop_label).rename({"msnt": "y"})

        self.__sch_pm_width_long_out_x.validate(x_pm_long.to_pandas(), lazy=True)
        self.__sch_pm_width_long_out_y.validate(y_pm_long.to_pandas(), lazy=True)

        # allocate the height to each x row
        
        long_widths_xy = x_pm_long.join(
            y_pm_long.select(pl.exclude("geoprop")),
            on=["p_idx", "peakprop"],
            how="inner",
        )

        self.__sch_pm_width_long_joined.validate(long_widths_xy.to_pandas(), lazy=True)
        
        self.widths_long_xy = long_widths_xy
        
        return self

    def _set_schemas(self):
        class PM_Width_In_X(pa.DataFrameModel):
            p_idx: int  # the peak idx
            whh_left: float
            whh_right: float
            pb_left: float
            pb_right: float

            class Config:
                name = "PM_WIdth_In_X"
                description = "the x values of the peak map data"
                strict = True

        self.__sch_pm_width_in_x = PM_Width_In_X

        class PM_Width_In_Y(pa.DataFrameModel):
            p_idx: int  # the peak idx
            whh_height: float
            pb_height: float

            class Config:
                name = "PM_Width_In_Y"
                description = "the y values of the widths"
                strict = True

        self.__sch_pm_width_in_y = PM_Width_In_Y

        class PM_Width_Long_Out_X(pa.DataFrameModel):
            """
            A generalized schema to verify that the x frame is as expected after
            melting.
            """

            p_idx: int = pa.Field(ge=0)  # peak idx
            peakprop: str = pa.Field(isin=["whh", "pb"])
            geoprop: str = pa.Field(isin=["left", "right"])
            x: float

            class Config:
                name = "PM_Width_Long_Out_X"
                strict = True

        self.__sch_pm_width_long_out_x = PM_Width_Long_Out_X

        class PM_Width_Long_Out_Y(pa.DataFrameModel):
            """
            A generalized schema to verify that the y frames are as expected after
            melting.
            """

            p_idx: int = pa.Field(
                ge=0,
            )  # peak idx
            peakprop: str = pa.Field(isin=["whh", "pb"])
            geoprop: str = pa.Field(isin=["height"])
            y: float

            class Config:
                name = "PM_Width_Long_Out_Y"
                strict = True

        self.__sch_pm_width_long_out_y = PM_Width_Long_Out_Y

        class PM_Width_Long_Joined(pa.DataFrameModel):
            p_idx: int = pa.Field(
                ge=0,
            )
            peakprop: str = pa.Field(
                isin=["whh", "pb"],
            )
            geoprop: str = pa.Field(isin=["left", "right"])
            x: float
            y: float

            class Config:
                name = "PM_Width_Long_Joined"
                strict = True

        self.__sch_pm_width_long_joined = PM_Width_Long_Joined


class WidthPlotter(PlotCore):
    """
    For handling all width plotting - WHH and bases. The core tenant is that we want to
    plot the widths peak by peak in such a way that it is clear which mark belongs to
    which peak, and which is left and which is right.
    """

    def __init__(self, ax: Axes, df: pd.DataFrame):
        super().__init__(ax, df)

    def plot_widths(
        self,
        y_colname: str,
        left_x_colname: str,
        right_x_colname: str,
        marker: str,
        label: str = "width",
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Main interface for plotting the widths. Wrap this in an outer function for specific measurements i.e. whh, bases
        """
        # shift the marker up enough to mark the location rather than occlude the signal

        # df_max = self.df[y_key].max()
        # self.df[y_key] = self.df[y_key] + df_max * 0.001

        self.check_keys_in_index([left_x_colname, right_x_colname], self.df.columns)

        for idx, s in self.df.iterrows():
            color: rgba = self.colors[idx]  # type: ignore

            self.ax.plot(
                [s[left_x_colname], s[right_x_colname]],
                [s[y_colname], s[y_colname]],
                c=color,
                marker=marker,
                markeredgecolor="black",
                ls="",
                label="_",
                **plot_kwargs,
            )

        self.add_proxy_artist_to_legend(label=label, line_2d=self.ax.lines[-1])

        return self
