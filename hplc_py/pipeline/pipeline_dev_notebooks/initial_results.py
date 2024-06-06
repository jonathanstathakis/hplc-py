"""
"""

from dataclasses import dataclass
import pandera as pa
import polars as pl
from pandera.typing.polars import DataFrame

from hplc_py.pipeline.preprocess_dashboard import PreProcesser

# Initial Results Tables Schemas


class Signals(pa.DataFrameModel):
    n_iter: int
    w_type: str
    w_idx: float
    time: float
    signal: float
    adjusted_signal: float
    background: float

    class Config:
        ordered = True
        strict = True
        coerce = True
        name = "Signals"


class Maxima(pa.DataFrameModel):
    n_iter: int
    p_idx: int
    loc: str
    dim: str
    value: float

    class Config:
        ordered = True
        strict = True
        name = "Maxima"
        coerce = True


class ContourLineBounds(pa.DataFrameModel):
    n_iter: int
    p_idx: int
    loc: str
    msnt: str
    dim: str
    value: float

    class Config:
        strict = True
        ordered = True
        coerce = True
        name = "ContourLineBounds"


class Widths(pa.DataFrameModel):
    n_iter: int
    p_idx: int
    msnt: str
    value: float

    class Config:
        name = "Widths"
        strict = True
        ordered = True
        coerce = True


class WindowBounds(pa.DataFrameModel):
    n_iter: int
    w_type: str
    w_idx: float
    start: float
    end: float

    class Config:
        name = "WindowBounds"
        ordered = True
        strict = True
        coerce = True


@dataclass
class Results:
    signals: DataFrame[Signals]
    maxima: DataFrame[Maxima]
    contour_line_bounds: DataFrame[ContourLineBounds]
    widths: DataFrame[Widths]
    window_bounds: DataFrame[WindowBounds]
    prepro_obj: dict[str, PreProcesser]


class N_Iter_Initial_Results:
    """
    Generate preprocessing results for a range of SNIP `n_iter`. Results are provided within a Results 
    container class which exposes the following tables: 'signals', 'maxima', 'contour_line_bounds', 'widths',
    'window_bounds', and a dict: 'prepro_obj'. Each is indexed by `n_iter`.
    
    Use by: 
    1. initialise class object with the `n_iter` range
    2. `load_dset`, providing the time and amplitude column keys
    3. `get_n_iter_results`, returns a Results container dataclass
    
    - signal (raw, background, adjusted, w_type, w_idx)
    - peak map
    - window bounds
    They then all need to be labeled as per the n_iter that generated them. Finally, they will all need to be collected, then appended together. 3 tables. A factory function outputting a container class of the three. Factory func takes the input data, n_iter bounds, sets up a container for each table, generates the results, labels the results as per the n_iter, concatenates the results, assigns the concatenated tables to a container class, returns the container class.
    """
    def __init__(self, n_iter_start: int, n_iter_end: int):        
        self.dset = pl.DataFrame()
        self.time_col = ""
        self.amp_col = ""
        self.experiments = []
        self.n_iter_start = n_iter_start
        self.n_iter_end = n_iter_end
        self.experiment_iterations = {}

    def load_dset(self, dset: pl.DataFrame, time_col: str, amp_col: str):
        self.dset = dset
        self.time_col = time_col
        self.amp_col = amp_col

    def get_n_iter_results(
        self,
    ) -> Results:
        """
        Iterating over the range between `n_iter_start` and `n_iter_end`, collect the:
            - signals
            - maxima
            - contour_line_bounds
            - widths
            - window_bounds
            - preprocessor object (for debugging)
        and concatenate all but 'preprocessor_obj' together in one table indexed by 'n_iter'.

        Preprocessor object is stored in a dict whose key is 'n_iter'
        Results are stored in a dataclass as per their name above.
        """

        # an intermediate container to collect each iteration results prior to concantenation
        msnts = {}

        for key in [
            "signals",
            "maxima",
            "contour_line_bounds",
            "widths",
            "window_bounds",
        ]:
            msnts[key] = []

        # prepro_obj will be stored differently, as a dict whose key is 'n_iter'

        msnts["prepro_obj"] = {}

        for n_iter in range(self.n_iter_start, self.n_iter_end + 1):
            prepro = PreProcesser()
            prepro.ingest_signal(
                signal=self.dset, time_col=self.time_col, amp_col=self.amp_col
            )
            prepro.signal_adjustment(bcorr__n_iter=n_iter)

            find_peaks_kwargs = {"prominence": 0.01}
            prepro.map_peaks(find_peaks_kwargs=find_peaks_kwargs)
            prepro.map_windows()

            # assign individual tables as polars
            tables = {}
            tables["signals"] = prepro.signal
            tables["contour_line_bounds"] = (
                prepro.peak_mapper.peak_map_.contour_line_bounds.pipe(pl.from_pandas)
            )
            tables["maxima"] = prepro.peak_mapper.peak_map_.maxima.pipe(pl.from_pandas)
            tables["widths"] = prepro.peak_mapper.peak_map_.widths.pipe(pl.from_pandas)
            tables["window_bounds"] = prepro.window_mapper.window_bounds_

            # add n_iter primary key column
            labeled_tables = {}
            for k, v in tables.items():
                labeled_tables[k] = v.insert_column(
                    index=0,
                    column=pl.Series(name="n_iter", values=[int(n_iter)] * len(v)),
                )

            # add the results to the msnts dict

            msnts["signals"].append(labeled_tables["signals"].pipe(DataFrame[Signals]))
            msnts["contour_line_bounds"].append(
                labeled_tables["contour_line_bounds"].pipe(DataFrame[ContourLineBounds])
            )
            msnts["maxima"].append(labeled_tables["maxima"].pipe(DataFrame[Maxima]))
            msnts["widths"].append(labeled_tables["widths"].pipe(DataFrame[Widths]))
            msnts["window_bounds"].append(
                labeled_tables["window_bounds"].pipe(DataFrame[WindowBounds])
            )

            # add 'prepro_obj' indexed to its 'n_iter' for future debugging
            msnts["prepro_obj"][str(n_iter)] = prepro

        # concatenate the results tables together

        concatenated_tables = {}

        for key, arr in msnts.items():
            # skip 'prepro_obj'
            if key == "prepro_obj":
                pass
            else:
                concatenated_tables[key] = pl.concat(arr, how="vertical")

        # now store results in the Results dataclass

        self.results_ = Results(
            signals=concatenated_tables["signals"],
            maxima=concatenated_tables["maxima"],
            contour_line_bounds=concatenated_tables["contour_line_bounds"],
            widths=concatenated_tables["widths"],
            window_bounds=concatenated_tables["window_bounds"],
            prepro_obj=msnts["prepro_obj"],
        )

        return self.results_
