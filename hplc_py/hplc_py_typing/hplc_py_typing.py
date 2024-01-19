import pandera.extensions as extensions
from hplc_py import P0AMP, P0TIME, P0WIDTH, P0SKEW

from typing_extensions import TypedDict
from typing import Any

import pandas as pd
import pandera as pa
from pandera.typing import Index, Series, DataFrame

import numpy.typing as npt
import numpy as np
from typing import Optional, TypeAlias

from hplc_py.hplc_py_typing.interpret_model import interpret_model

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
pdInt: TypeAlias = pd.Int64Dtype
pdFloat: TypeAlias = pd.Float64Dtype
rgba: TypeAlias = tuple[float, float, float, float]


@extensions.register_check_method(statistics=["col", "stats"])  # type: ignore
def check_stats(df, *, col: str, stats: dict):
    """
    Test basic statistics of a dataframe. Ideal for validating data in large frames.

    Provide stats as a dict of {'statistic_name' : expected_val}.
    Currently tested for: count, mean, std, min, max
    """
    # statistics that I have checked to behave as expected
    valid_stats = ["count", "mean", "std", "min", "max"]

    # validate keys
    for key in stats.keys():
        if not key in valid_stats:
            raise ValueError(f"{key} is not a valid statistic, please re-enter")

    # want to lazy validate so user gets the full picture, hence calculate all then check later
    checks = {}
    col_stats = {}
    for stat, val in stats.items():
        col_stat = df[col].agg(stat)

        checks[stat] = col_stat == val

        col_stats[stat] = col_stat

    # check all results, generating a report string for failures only then raising a Value Error
    error_strs = []
    if not all(checks.values()):
        for stat, test in checks.items():
            if not test:
                error_str = f"{col} has failed {stat} check. Expected {stat} is {stats[stat]}, but {col} {stat} is {col_stats[stat]}"
                error_strs.append(error_str)
        raise ValueError("\n" + "\n".join(error_strs))
    else:
        # if all checks pass, move on
        return True


from pandera.api.pandas.model_config import BaseConfig


class BaseDF(pa.DataFrameModel):
    """
    Lowest level class for basic DataFrame assumptions - for example, they will all
    contain a index named 'idx' which is the default RangedIndex
    """

    idx: Index[int] = pa.Field(check_name=True)

    @pa.check(
        "idx", name="idx_check", error=f"expected range index bounded by 0 and len(df)"
    )
    def check_is_range_index(cls, idx: Series[int]) -> bool:
        left_idx = pd.RangeIndex(0, len(idx) - 1)
        right_idx = pd.RangeIndex(idx.iloc[0], idx.iloc[-1])
        check = left_idx.equals(right_idx)
        return check

    class Config(BaseConfig):
        strict = True
        ordered = False


class InterpModelKwargs(TypedDict, total=False):
    schema_name: str
    inherit_from: str
    is_base: bool
    check_dict: dict
    pandas_dtypes: bool


def schema_tests(
    base_schema,
    dset_schema,
    base_schema_kwargs: InterpModelKwargs,
    dset_schema_kwargs: InterpModelKwargs,
    df,
    verbose: bool = False,
):
    base_schema_str = interpret_model(
        df,
        **base_schema_kwargs,
    )

    dset_schema_str = interpret_model(
        df,
        **dset_schema_kwargs,
    )
    try:
        base_schema(df)
    except Exception as e:
        print("")

        print(base_schema_str)
        raise ValueError("failed base schema test with error: " + str(e))

    try:
        dset_schema(df)
    except Exception as e:
        print("")

        print(dset_schema_str)
        raise ValueError(
            f"failed dataset schema test with error: {e}\n Printing schema for the input df.."
        )

    if verbose:
        print("## Base Schema ##\n\n")
        print(base_schema_str)
        print("\n")
        print("## Dset Schema ## \n\n")
        print(dset_schema_str)
        print("\n")
    return None


class SignalDFInBase(pa.DataFrameModel):
    """
    The base signal, with time and amplitude directions
    """

    tbl_name: Optional[str] = pa.Field(eq="testsignal")
    time: np.float64
    amp_raw: np.float64

    class Config:
        strict = True


class SignalDFInAssChrom(SignalDFInBase):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    tbl_name: np.object_ = pa.Field(isin=["testsignal"])
    time: np.float64 = pa.Field()
    amp_raw: np.float64 = pa.Field()

    class Config:
        name = "SignalDFInAssChrom"

        _time_basic_stats = {
            "col": "time",
            "stats": {
                "count": 15000.0,
                "min": 0.0,
                "max": 149.99,
                "mean": 74.995,
                "std": 43.30271354083945,
            },
        }
        _amp_raw_basic_stats = {
            "col": "amp_raw",
            "stats": {
                "count": 15000.0,
                "min": -0.0298383947260937,
                "max": 42.69012166052291,
                "mean": 2.1332819642622307,
                "std": 6.893591961394714,
            },
        }

        check_stats = _time_basic_stats
        check_stats = _amp_raw_basic_stats


class SignalDF(pa.DataFrameModel):
    time_idx: pd.Int64Dtype = pa.Field(coerce=False)
    time: pd.Float64Dtype = pa.Field(coerce=False)
    amp_raw: pd.Float64Dtype = pa.Field(coerce=False)
    amp_corrected: Optional[pd.Float64Dtype] = pa.Field(coerce=False)
    amp_bg: Optional[pd.Float64Dtype] = pa.Field(coerce=False)
    amp_corrected_norm: Optional[pd.Float64Dtype] = pa.Field(coerce=False)
    amp_norm: Optional[pd.Float64Dtype] = pa.Field(coerce=False)

    class Config:
        strict = True


class OutSignalDF_ManyPeaks(SignalDF):
    time_idx: np.int64 = pa.Field(ge=0, lt=7000)
    amp_raw: np.float64 = pa.Field(ge=6.425446e-48, le=3.989453e01)
    norm_amp: np.float64 = pa.Field(ge=0, le=1)


class OutSignalDF_AssChrom(SignalDF):
    time_idx: pd.Int64Dtype = pa.Field(
        coerce=False, in_range={"min_value": 0, "max_value": 149999}
    )
    time: pd.Float64Dtype = pa.Field(
        coerce=False, in_range={"min_value": 0, "max_value": 149.99}
    )
    amp_raw: pd.Float64Dtype = pa.Field(
        coerce=False,
        in_range={"min_value": -0.0298383947260937, "max_value": 42.69012166052291},
    )
    amp_corrected: Optional[pd.Float64Dtype] = pa.Field(
        coerce=False, in_range={"min_value": 0.007597293, "max_value": 42.703030473}
    )
    amp_bg: Optional[pd.Float64Dtype] = pa.Field(
        coerce=False,
        in_range={"min_value": -0.0075972928921949, "max_value": 0.0002465850460692323},
    )
    amp_corrected_norm: Optional[pd.Float64Dtype] = pa.Field(
        coerce=False, in_range={"min_value": 0, "max_value": 1}
    )
    amp_norm: Optional[pd.Float64Dtype] = pa.Field(
        coerce=False, in_range={"min_value": 0, "max_value": 1}
    )


class OutWindowDF_Base(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    time_idx: pd.Int64Dtype = pa.Field()
    window_idx: pd.Int64Dtype = pa.Field()
    window_type: pd.StringDtype = pa.Field()
    sw_idx: pd.Int64Dtype = pa.Field()

    class Config:
        name = "OutWindowDF_Base"
        strict = True


class OutWindowDF_AssChrom(OutWindowDF_Base):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    time_idx: pd.Int64Dtype = pa.Field()
    window_idx: pd.Int64Dtype = pa.Field()
    window_type: pd.StringDtype = pa.Field(isin=["interpeak", "peak"])
    sw_idx: pd.Int64Dtype = pa.Field()

    class Config:
        name = "OutWindowDF_AssChrom"
        strict = True

        _time_idx_basic_stats = {
            "col": "time_idx",
            "stats": {
                "count": 15000.0,
                "min": 0.0,
                "max": 14999.0,
                "mean": 7499.5,
                "std": 4330.271354083945,
            },
        }
        _window_idx_basic_stats = {
            "col": "window_idx",
            "stats": {
                "count": 15000.0,
                "min": 1.0,
                "max": 3.0,
                "mean": 1.5542,
                "std": 0.8244842903029327,
            },
        }
        _sw_idx_basic_stats = {
            "col": "sw_idx",
            "stats": {
                "count": 15000.0,
                "min": 1.0,
                "max": 5.0,
                "mean": 2.8232,
                "std": 1.3161271111503048,
            },
        }

        check_stats = _time_idx_basic_stats
        check_stats = _window_idx_basic_stats
        check_stats = _sw_idx_basic_stats


class P0(BaseDF):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    window_idx: pd.Int64Dtype = pa.Field()
    p_idx: pd.Int64Dtype = pa.Field()
    param: pd.CategoricalDtype = pa.Field(isin=[P0AMP, P0TIME, P0WIDTH, P0SKEW])
    p0: pd.Float64Dtype = pa.Field()

    class Config:
        name = "p0"
        strict = True
        coerce = False


class OutInitialGuessAssChrom(P0):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    window_idx: pd.Int64Dtype = pa.Field()
    peak_idx: pd.Int64Dtype = pa.Field()
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    p0: pd.Float64Dtype = pa.Field()

    class Config:
        name = "OutInitialGuessAssChrom"
        strict = True

        _window_idx_basic_stats = {
            "col": "window_idx",
            "stats": {
                "count": 16.0,
                "min": 1.0,
                "max": 2.0,
                "mean": 1.25,
                "std": 0.4472135954999579,
            },
        }
        _peak_idx_basic_stats = {
            "col": "peak_idx",
            "stats": {
                "count": 16.0,
                "min": 0.0,
                "max": 3.0,
                "mean": 1.5,
                "std": 1.1547005383792515,
            },
        }
        _p0_basic_stats = {
            "col": "p0",
            "stats": {
                "count": 16.0,
                "min": 0.0,
                "max": 110.0,
                "mean": 21.010537952826958,
                "std": 32.516266671353925,
            },
        }

        check_stats = _window_idx_basic_stats
        check_stats = _peak_idx_basic_stats
        check_stats = _p0_basic_stats


class Bounds(BaseDF):
    window_idx: pd.Int64Dtype
    p_idx: pd.Int64Dtype
    param: pd.CategoricalDtype
    lb: pd.Float64Dtype = pa.Field(nullable=False)
    ub: pd.Float64Dtype = pa.Field(nullable=False)


class BoundsAssChrom(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    window_idx: pd.Int64Dtype = pa.Field(
        eq=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    )
    peak_idx: pd.Int64Dtype = pa.Field(
        eq=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    )
    param: pd.CategoricalDtype = pa.Field(
        eq=[
            "amp",
            "loc",
            "skew",
            "whh",
            "amp",
            "loc",
            "skew",
            "whh",
            "amp",
            "loc",
            "skew",
            "whh",
            "amp",
            "loc",
            "skew",
            "whh",
        ]
    )
    lb: pd.Float64Dtype = pa.Field(
        eq=[
            4.2687835886729895,
            6.87,
            -np.inf,
            0.01,
            1.9957724534443004,
            6.87,
            -np.inf,
            0.01,
            0.26518058154648144,
            6.87,
            -np.inf,
            0.01,
            3.9886630747203804,
            102.78,
            -np.inf,
            0.01,
        ]
    )
    ub: pd.Float64Dtype = pa.Field(
        eq=[
            426.8783588672989,
            99.31,
            np.inf,
            46.22,
            199.57724534443003,
            99.31,
            np.inf,
            46.22,
            26.518058154648145,
            99.31,
            np.inf,
            46.22,
            398.86630747203805,
            117.54,
            np.inf,
            7.380000000000003,
        ]
    )

    class Config:
        name = "OutDefaultBoundsAssChrom"
        strict = True


class WindowedSignalAssChrom(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    time_idx: np.int64 = pa.Field()
    time: np.float64 = pa.Field()
    amp_raw: np.float64 = pa.Field()
    amp_corrected: np.float64 = pa.Field()
    amp_bg: np.float64 = pa.Field()
    sw_idx: pd.Int64Dtype = pa.Field()
    window_type: pd.StringDtype = pa.Field(isin=["interpeak", "peak"])
    window_idx: pd.Int64Dtype = pa.Field()

    class Config:
        name = "OutWindowedSignalAssChrom"
        strict = True

        _time_idx_basic_stats = {
            "col": "time_idx",
            "stats": {
                "count": 15000.0,
                "min": 0.0,
                "max": 14999.0,
                "mean": 7499.5,
                "std": 4330.271354083945,
            },
        }
        _time_basic_stats = {
            "col": "time",
            "stats": {
                "count": 15000.0,
                "min": 0.0,
                "max": 149.99,
                "mean": 74.995,
                "std": 43.30271354083945,
            },
        }
        _amp_raw_basic_stats = {
            "col": "amp_raw",
            "stats": {
                "count": 15000.0,
                "min": -0.0298383947260937,
                "max": 42.69012166052291,
                "mean": 2.1332819642622307,
                "std": 6.893591961394714,
            },
        }
        _amp_corrected_basic_stats = {
            "col": "amp_corrected",
            "stats": {
                "count": 15000.0,
                "min": -0.0298383947260937,
                "max": 42.68783588672989,
                "mean": 2.126953612030828,
                "std": 6.894150253857469,
            },
        }
        _amp_bg_basic_stats = {
            "col": "amp_bg",
            "stats": {
                "count": 15000.0,
                "min": 0.0,
                "max": 0.007843877938264132,
                "mean": 0.006328352231402279,
                "std": 0.0022581116429225144,
            },
        }
        _sw_idx_basic_stats = {
            "col": "sw_idx",
            "stats": {
                "count": 15000.0,
                "min": 1.0,
                "max": 5.0,
                "mean": 2.8232,
                "std": 1.3161271111503048,
            },
        }
        _window_idx_basic_stats = {
            "col": "window_idx",
            "stats": {
                "count": 15000.0,
                "min": 1.0,
                "max": 3.0,
                "mean": 1.5542,
                "std": 0.8244842903029327,
            },
        }

        check_stats = _time_idx_basic_stats
        check_stats = _time_basic_stats
        check_stats = _amp_raw_basic_stats
        check_stats = _amp_corrected_basic_stats
        check_stats = _amp_bg_basic_stats
        check_stats = _sw_idx_basic_stats
        check_stats = _window_idx_basic_stats


@extensions.register_check_method(statistics=["col", "col_lb"])
def col_in_lb(df, *, col: str, col_lb: str):
    """
    Test if p0 is in lower bounds. Needed for `curve_fit`
    """

    return df[col] > df[col_lb]


@extensions.register_check_method(statistics=["col", "col_ub"])
def col_in_ub(df, *, col: str, col_ub: str):
    """
    Test if p0 is in upper bounds. Needed for `curve_fit`
    """
    return df[col] < df[col_ub]


class Params(Bounds, P0):
    pass

    class Config(BaseConfig):
        col_in_lb = {"col": "p0", "col_lb": "lb"}
        col_in_ub = {"col": "p0", "col_ub": "ub"}


class OutParamManyPeaks(Params):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    p0: pd.Float64Dtype = pa.Field(
        isin=[
            0.0,
            39.89452538404621,
            110.0,
            39.89437671209474,
            1000.0,
            110.5,
            1500.0,
            2000.0,
            2500.0,
            3000.0,
            3500.0,
            4000.0,
            4500.0,
            5000.0,
            5500.0,
            117.5,
        ]
    )
    lb: pd.Float64Dtype = pa.Field(
        isin=[636.0, 1.0, -np.inf, 3.9894525384046213, 3.9894376712094743]
    )
    ub: pd.Float64Dtype = pa.Field(
        isin=[5868.0, 2616.0, np.inf, 398.9452538404621, 398.9437671209474]
    )
    inbounds: bool = pa.Field(isin=[True])


class OutParamsAssChrom(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    window_idx: pd.Int64Dtype = pa.Field(
        eq=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    )
    peak_idx: pd.Int64Dtype = pa.Field(
        eq=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    )
    param: pd.CategoricalDtype = pa.Field(
        eq=[
            "amp",
            "loc",
            "whh",
            "skew",
            "amp",
            "loc",
            "whh",
            "skew",
            "amp",
            "loc",
            "whh",
            "skew",
            "amp",
            "loc",
            "whh",
            "skew",
        ]
    )
    p0: pd.Float64Dtype = pa.Field(
        eq=[
            42.68783588672989,
            15.07,
            1.352563701131644,
            0.0,
            19.957724534443003,
            18.99,
            0.8626565883964099,
            0.0,
            2.6518058154648143,
            80.0,
            3.531976828454367,
            0.0,
            39.8866307472038,
            110.0,
            1.1774131434073842,
            0.0,
        ]
    )
    lb: pd.Float64Dtype = pa.Field(
        eq=[
            4.2687835886729895,
            6.87,
            0.01,
            -np.inf,
            1.9957724534443004,
            6.87,
            0.01,
            -np.inf,
            0.26518058154648144,
            6.87,
            0.01,
            -np.inf,
            3.9886630747203804,
            102.78,
            0.01,
            -np.inf,
        ]
    )
    ub: pd.Float64Dtype = pa.Field(
        eq=[
            426.8783588672989,
            99.31,
            46.22,
            np.inf,
            199.57724534443003,
            99.31,
            46.22,
            np.inf,
            26.518058154648145,
            99.31,
            46.22,
            np.inf,
            398.86630747203805,
            117.54,
            7.380000000000003,
            np.inf,
        ]
    )

    class Config:
        name = "OutParamsAssChrom"
        strict = True


class Popt(BaseDF):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    window_idx: pd.Int64Dtype = pa.Field()
    p_idx: pd.Int64Dtype = pa.Field()
    amp: pd.Float64Dtype = pa.Field()
    time: pd.Float64Dtype = pa.Field()
    whh_half: pd.Float64Dtype = pa.Field()
    skew: pd.Float64Dtype = pa.Field()

    class Config(BaseConfig):
        name = "Popt"
        strict = True


class OutPoptDF_AssChrom(Popt):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    tbl_name: pd.StringDtype = pa.Field(eq=["popt", "popt", "popt", "popt"])
    window_idx: pd.Int64Dtype = pa.Field(eq=[1, 1, 1, 2])
    peak_idx: pd.Int64Dtype = pa.Field(eq=[0, 1, 2, 3])
    amp: pd.Float64Dtype = pa.Field(
        eq=[100.01095135529069, 99.95293326727652, 19.87594297337034, 99.95961060435313]
    )
    loc: pd.Float64Dtype = pa.Field(
        eq=[14.907132797396796, 18.99976077649315, 79.95550926291489, 110.0051788176931]
    )
    whh: pd.Float64Dtype = pa.Field(
        eq=[
            1.0043771976008562,
            1.9992972533746571,
            2.9879782131837924,
            0.9997441651276472,
        ]
    )
    skew: pd.Float64Dtype = pa.Field(
        eq=[
            0.11692924538584329,
            0.00028775901715210814,
            0.018674051548999857,
            -0.0064925018307393475,
        ]
    )

    class Config:
        name = "OutPoptDF_AssChrom"
        strict = True


def isArrayLike(x: Any):
    if not any(x):
        raise ValueError("x is None")

    if not hasattr(x, "__array__"):
        return False
    else:
        return True


class Recon(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    peak_idx: pd.Int64Dtype = pa.Field()
    time_idx: pd.Int64Dtype = pa.Field()
    time: np.float64 = pa.Field()
    unmixed_amp: np.float64 = pa.Field()

    class Config:
        name = "OutReconDFBase"
        strict = True


class OutReconDF_AssChrom(Recon):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    peak_idx: pd.Int64Dtype = pa.Field()
    time: np.float64 = pa.Field()
    unmixed_amp: np.float64 = pa.Field()

    class Config:
        name = "OutReconDF_AssChrom"
        strict = True

        _peak_idx_basic_stats = {
            "col": "peak_idx",
            "stats": {
                "count": 60000.0,
                "min": 0.0,
                "max": 3.0,
                "mean": 1.5,
                "std": 1.1180433058162647,
            },
        }
        _time_basic_stats = {
            "col": "time",
            "stats": {
                "count": 60000.0,
                "min": 0.0,
                "max": 149.99,
                "mean": 74.995,
                "std": 43.30163094142494,
            },
        }
        _unmixed_amp_basic_stats = {
            "col": "unmixed_amp",
            "stats": {
                "count": 60000.0,
                "min": 0.0,
                "max": 39.89647928924097,
                "mean": 0.5329990636671511,
                "std": 3.395449541122017,
            },
        }

        check_stats = _peak_idx_basic_stats
        check_stats = _time_basic_stats
        check_stats = _unmixed_amp_basic_stats


class OutPeakReportBase(Popt):
    tbl_name: pd.StringDtype = pa.Field(eq="peak_report")
    retention_time: pd.Float64Dtype
    unmixed_area: pd.Float64Dtype
    unmixed_maxima: pd.Float64Dtype


class OutPeakReportAssChrom(OutPeakReportBase):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    tbl_name: pd.StringDtype = pa.Field(
        eq=["peak_report", "peak_report", "peak_report", "peak_report"]
    )
    window_idx: pd.Int64Dtype = pa.Field(eq=[1, 1, 1, 2])
    peak_idx: pd.Int64Dtype = pa.Field(eq=[0, 1, 2, 3])
    retention_time: pd.Float64Dtype = pa.Field(
        eq=[
            0.14907132797396797,
            0.18999760776493152,
            0.7995550926291489,
            1.1000517881769312,
        ]
    )
    loc: pd.Float64Dtype = pa.Field(
        eq=[14.907132797396796, 18.99976077649315, 79.95550926291489, 110.0051788176931]
    )
    amp: pd.Float64Dtype = pa.Field(
        eq=[100.01095135529069, 99.95293326727652, 19.87594297337034, 99.95961060435313]
    )
    whh: pd.Float64Dtype = pa.Field(
        eq=[
            1.0043771976008562,
            1.9992972533746571,
            2.9879782131837924,
            0.9997441651276472,
        ]
    )
    skew: pd.Float64Dtype = pa.Field(
        eq=[
            0.11692924538584329,
            0.00028775901715210814,
            0.018674051548999857,
            -0.0064925018307393475,
        ]
    )
    unmixed_area: np.float64 = pa.Field(
        eq=[10001.095135529069, 9995.293326727651, 1987.594297337034, 9995.961060435313]
    )
    unmixed_maxima: np.float64 = pa.Field(
        eq=[
            39.89647928924097,
            19.944734017503087,
            2.6540468066889553,
            39.88885501981935,
        ]
    )

    class Config:
        name = "OutPeakReportAssChrom"
        strict = True
