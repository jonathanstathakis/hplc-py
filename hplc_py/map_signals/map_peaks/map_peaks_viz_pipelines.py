import abc
from typing import Self

import pandera as pa
import polars as pl
from pandera.typing import DataFrame

from hplc_py.hplc_py_typing.hplc_py_typing import PeakMap
from hplc_py.io_validation import IOValid

from .map_peaks_viz_schemas import (
    Maxima_X_Y,
    PM_Width_In_X,
    PM_Width_In_Y,
    PM_Width_Long_Joined,
    PM_Width_Long_Out_X,
    PM_Width_Long_Out_Y,
    Width_Maxima_Join,
)


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
            self._pipe_peak_widths_to_long.load_pipeline(
                self.peak_map.to_pandas().rename_axis(index="idx")
            )
            .run_pipeline()
            .widths_long_xy
        )

        maxima_x_y = (
            self._pipe_peak_maxima_to_long.load_pipeline(
                self.peak_map.to_pandas().rename_axis(index="idx")
            )
            .run_pipeline()
            .maxima_x_y
        )

        self.peak_map_plot_data = (
            self._pipe_join_width_maxima_long.load_pipeline(
                widths_long_xy=widths_long_xy.to_pandas(),
                maxima_x_y=maxima_x_y.to_pandas(),
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
