from typing import Self
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
from .map_peaks_viz_schemas import (
    PM_Width_In_X,
    PM_Width_In_Y,
    PM_Width_Long_Out_X,
    PM_Width_Long_Out_Y,
    PM_Width_Long_Joined,
    Maxima_X_Y,
    Width_Maxima_Join,
)


class PlotCore(IOValid):
    """
    Base Class of the peak map plots, containing style settings and common methods.
    """

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

        wp.plot_widths_vertices(
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

        wp.plot_widths_vertices(
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


class Pipeline(metaclass=abc.ABCMeta):
    """
    A basic template for pipelines - objects that take one or more inputs and produce
    one output, with internal validation via Pandera DataFrameModel schemas.

    A sklearn inspired Pipeline object which is used by first running `load_pipeline`,
    then `run_pipeline` and accessing the desired output. Integrates Pandera schemas
    for input, intermediate and output validation.
    """

    @abc.abstractmethod
    def __init__(self):
        self._set_internal_schemas()

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
    def _set_internal_schemas(self):
        """
        Initialise defined schemas as objects of self.
        """
        raise NotImplementedError


class Pipeline_Peak_Map_Interface(Pipeline, IOValid):
    """
    Combines several pipelines to produce a long frame indexed by the peak index
    'p_idx', peak prop ('whh','pb') and geoprop ('left', 'right'). This format is
    useful for visualising the left and right ips calculated by `scipy.peak_widths`
    """

    def __init__(self):
        """
        Initialise peak map and X as polars if not already. Up to the user to ensure that nothing lost in transition. Preference is to pass a polars frame rather than pandas.
        """

        self._peak_map = pl.DataFrame()
        self.peak_map_plot_data = pl.DataFrame()

        self._pipe_peak_widths_to_long = Pipe_Peak_Widths_To_Long()
        self._pipe_peak_maxima_to_long = Pipe_Peak_Maxima_To_Long()
        self._pipe_join_width_maxima_long = Pipeline_Join_Width_Maxima_Long()

    @pa.check_types
    def load_pipeline(self, peak_map: DataFrame[PeakMap]) -> Self:
        if not self._is_polars(peak_map):
            self.peak_map = pl.from_pandas(peak_map)
        else:
            self.peak_map = peak_map

        return self

    def run_pipeline(self) -> Self:

        widths_long_xy = (
            self._pipe_peak_widths_to_long.load_pipeline(self.peak_map.to_pandas().rename_axis(index='idx'))
            .run_pipeline()
            .widths_long_xy
        )

        maxima_x_y = (
            self._pipe_peak_maxima_to_long.load_pipeline(self.peak_map.to_pandas().rename_axis(index='idx'))
            .run_pipeline()
            .maxima_x_y
        )

        self.peak_map_plot_data = (
            self._pipe_join_width_maxima_long.load_pipeline(
                widths_long_xy=widths_long_xy.to_pandas(), maxima_x_y=maxima_x_y.to_pandas()
            )
            .run_pipeline()
            .width_maxima_join
        )
        return self

    def _set_internal_schemas(self):
        pass

        return self


class Pipe_Peak_Widths_To_Long(Pipeline, IOValid):
    """
    Extract the peak width properties from `peak_map` and arrange them in long form.

    This pipe produces one output `widths_long_xy`.
    """

    def __init__(self):
        self._set_internal_schemas()

    @pa.check_types(lazy=True)
    def load_pipeline(
        self,
        peak_map: DataFrame[PeakMap],
        idx_cols: list[str] = ["p_idx"],
        x_cols: list[str] = ["whh_left", "whh_right", "pb_left", "pb_right"],
        y_cols: list[str] = ["whh_height", "pb_height"],
    ) -> Self:
        # internal logic uses polars, but currently pandera does not support them.
        if not self._is_polars(peak_map):
            self._peak_map = pl.from_pandas(peak_map)
        else:
            self._peak_map = peak_map

        self._idx_cols = idx_cols
        self._x_cols = x_cols
        self._y_cols = y_cols

        self.widths_long_xy: pl.DataFrame = pl.DataFrame()

        return self

    def run_pipeline(
        self,
    ) -> Self:
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

        def melt_peak_prop_label(df: pl.DataFrame):
            out = (
                df.melt(id_vars="p_idx", variable_name="peak_prop", value_name="msnt")
                .with_columns(pl.col("peak_prop").str.split(by="_").list.to_struct())
                .unnest("peak_prop")
                .rename({"field_0": "peak_prop", "field_1": "geoprop"})
            )

            return out

        x_pm_long = x_pm.pipe(melt_peak_prop_label).rename({"msnt": "x"})
        y_pm_long = y_pm.pipe(melt_peak_prop_label).rename({"msnt": "y"})

        self.__sch_pm_width_long_out_x.validate(x_pm_long.to_pandas(), lazy=True)
        self.__sch_pm_width_long_out_y.validate(y_pm_long.to_pandas(), lazy=True)

        # allocate the height to each x row

        long_widths_xy = x_pm_long.join(
            y_pm_long.select(pl.exclude("geoprop")),
            on=["p_idx", "peak_prop"],
            how="inner",
        )

        self.__sch_pm_width_long_joined.validate(long_widths_xy.to_pandas(), lazy=True)

        self.widths_long_xy = long_widths_xy

        return self

    def _set_internal_schemas(self):
        self.__sch_pm_width_in_x = PM_Width_In_X

        self.__sch_pm_width_in_y = PM_Width_In_Y

        self.__sch_pm_width_long_out_x = PM_Width_Long_Out_X

        self.__sch_pm_width_long_out_y = PM_Width_Long_Out_Y

        self.__sch_pm_width_long_joined = PM_Width_Long_Joined


class Pipe_Peak_Maxima_To_Long(Pipeline, IOValid):
    def __init__(self):
        self._set_internal_schemas()

        self.peak_map: pl.DataFrame = pl.DataFrame()
        self.maxima_x_y: pl.DataFrame = pl.DataFrame()

    @pa.check_types
    def load_pipeline(
        self,
        peak_map: DataFrame[PeakMap],
    ) -> Self:
        if not self._is_polars(peak_map):
            self.peak_map = pl.from_pandas(peak_map)
        else:
            self.peak_map = peak_map

        return self

    def run_pipeline(
        self, idx_colnames=["p_idx"], x_colname="X_idx", y_colname="maxima"
    ) -> Self:
        """
        A straight forward selection from the greater `peak_map`, selecting the 'p_idx',
        and x and y of the peak maxima points as 'maxima_x', and 'maxima_y',
        respectively. Returns self, access attribute `maxima_x_y` to obtain the result

        :param idx_colnames: the peak index column of `peak_map`, defaults to ['p_idx']
        :type idx_colnames: [list['str'], optional]
        :param x_colname: the name of the column containing the x values of the peak
        maxima, defaults to 'X_idx'
        :type x_colname: str, optional
        :param y_colname: the name of the column containing the y values of the peak
        maxima, defaults to 'maxima'
        :type y_colname: str, optional
        :return: self. access attribute `maxima_x_y` to obtain the result.
        :rtype: Self
        """

        maxima_x_y = self.peak_map.select(
            pl.col(idx_colnames + [x_colname, y_colname])
        ).rename({x_colname: "maxima_x", y_colname: "maxima_y"})

        self.__sch_maxima_x_y.validate(maxima_x_y.to_pandas(), lazy=True)

        self.maxima_x_y = maxima_x_y
        return self

    def _set_internal_schemas(self) -> DataFrame[Maxima_X_Y]:
        self.__sch_maxima_x_y = Maxima_X_Y


class Pipeline_Join_Width_Maxima_Long(Pipeline, IOValid):
    """
    Produce a table for plotting a line between the base width ips and the maxima to
    describe the mapping of the peak. Each row needs an x1, y1, x2, y2, and will
    labeled as the property of the base, i.e. line 1 will be labeled pb left, with
    the maxima values being x2, y2. Rename them in this pipeline. Joining on `p_idx`.

    After running `load_pipeline().run_pipeline()` access the result in `width_maxima_join`
    """

    def __init__(self):
        self._set_internal_schemas()

        self.widths_long_xy: pl.DataFrame = pl.DataFrame()
        self.maxima_x_y: pl.DataFrame = pl.DataFrame()
        self.width_maxima_join: pl.DataFrame = pl.DataFrame()

    @pa.check_types
    def load_pipeline(
        self,
        widths_long_xy: DataFrame[PM_Width_Long_Joined],
        maxima_x_y: DataFrame[Maxima_X_Y],
    ) -> Self:
        """
        :param widths_long_xy: frame produced by `Pipeline_Peak_Widths_To_Long`, a long
        form of the `peak_map`
        :type widths_long_xy: DataFrame[PM_Width_Long_Joined]
        :param maxima_x_y: frame produced by `Pipe_Peak_Maxima_To_Long`, the peak maxima
        values in `peak_map`
        :type maxima_x_y: DataFrame[Maxima_X_Y]
        :return: self
        :rtype: Self
        """

        if not self._is_polars(widths_long_xy):
            self.widths_long_xy = pl.from_pandas(widths_long_xy)
        else:
            self.widths_long_xy = widths_long_xy

        if not self._is_polars(maxima_x_y):
            self.maxima_x_y = pl.from_pandas(maxima_x_y)
        else:
            self.maxima_x_y = maxima_x_y

        return self

    def run_pipeline(self):
        """
        Inner join of the width and maxima frames on 'p_idx', renames 'x' as 'x1',
        'y' as 'y1', 'x2', 'maxima_x' as 'x2', 'maxma_y' as 'y2'
        """
        self.width_maxima_join: pl.DataFrame = self.widths_long_xy.join(
            self.maxima_x_y, on="p_idx", how="inner", validate="m:1"
        ).rename({"x": "x1", "y": "y1", "maxima_x": "x2", "maxima_y": "y2"})

        Width_Maxima_Join.validate(self.width_maxima_join.to_pandas(), lazy=True)

        return self

    def _set_internal_schemas(self) -> None:
        pass


class WidthPlotter(PlotCore):
    """
    For handling all width plotting - WHH and bases. The core tenant is that we want to
    plot the widths peak by peak in such a way that it is clear which mark belongs to
    which peak, and which is left and which is right.

    Provides an option for plotting markers at the ips `plot_widths_vertices`, and
    lines connecting the ips to the maxima in `plot_widths_edges`
    """

    def __init__(self, ax: Axes, df: pd.DataFrame):
        super().__init__(ax, df)

    def plot_widths_vertices(
        self,
        y_colname: str,
        left_x_colname: str,
        right_x_colname: str,
        marker: str,
        label: str = "width",
        plot_kwargs: dict = {},
    ) -> Self:
        """
        Main interface for plotting the width ips as points.
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

    def plot_widths_edges(
        self,
        width_maxima_join: DataFrame[Width_Maxima_Join],
    ):
        pass

        return self
