from typing import Any

import pandas as pd
import pandera as pa
import pandera.typing as pt
import pandera.extensions as extensions

import numpy.typing as npt
import numpy as np
from typing import Optional


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


def interpret_model(
    df,
    schema_name=None,
    inherit_from: str = "",
    is_base: bool = False,
    check_dict: dict = {},
):
    """
    Output a string representation of a dataframe schema DataFrameModel with datatypes and checks.
    Outputs both a base model and a specific model.

    check_dict: check type for each column. specify column name as key, check type: 'eq', 'isin', 'basic_stats'. if no checkdict is passed and is_base is false, generate an opinionated check dict based on types and dataframe size. use 'eq' if frame is less than 10 rows long, else use 'basic_stats' and 'isin'. Problem with this compression approach is that you lose information on ordering.

    """
    custom_indent_mag = 1
    indent_str = " "
    base_indent_mag = 4
    base_indent = "".join([indent_str] * base_indent_mag)
    indents = "".join(base_indent * custom_indent_mag)
    df_columns = df.columns

    # assert no check_dict passed if is_base is True

    if (is_base) and (check_dict):
        raise ValueError("do not provide a check_dict if is_base == True")

    def eq_checkstr(series):
        return f"eq={series.tolist()}".replace("inf", "np.inf")

    def isin_checkstr(series):
        return f"isin={series.unique().tolist()}".replace("inf", "np.inf")

    if is_base:
        check_dict = {col: "" for col in df_columns}

    # generate opinionated default check_dict
    
    if not check_dict and not is_base:
        
        if len(df)<=10:
            check_dict = {col: "eq" for col in df}
        else:
            check_dict = {col: "basic_stats" if pd.api.types.is_numeric_dtype(df[col]) else "isin" for col in df}
            
    # generate the check strings
    
    check_strs = {}

    gen_basic_stats = "basic_stats" in check_dict.values()
    if gen_basic_stats:
        basic_stats = ["count", "min", "max", "mean", "std"]
        basic_stats_dicts = {}

    # prepare the checks

    for col, check_str in check_dict.items():
        series = df[col]
        if check_str == "eq":
            check_strs[col] = eq_checkstr(series)
        elif check_str == "isin":
            check_strs[col] = isin_checkstr(series)
        elif check_str == "basic_stats":
            # must be a numerical column
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"{col} must be numeric to use 'basic_stats' option")
            else:
                basic_stats_dicts[col] = dict(
                    zip(basic_stats, series.describe()[basic_stats].to_list())
                )

    # define datatypes with appending/preppending if necessary for imports
    dtypes = {}
    for col in df_columns:
        dt = str(df[col].dtype)

        amended_dtype = None

        if any(str.isupper(c) for c in dt):
            amended_dtype = "pd." + dt + "Dtype"
        elif any(str.isnumeric(c) for c in dt):
            amended_dtype = "np." + dt
        elif dt == "object":
            amended_dtype = "np.object_"
        elif dt == "category":
            amended_dtype = "pd.CategoricalDtype"
        if amended_dtype:
            dtypes[col] = amended_dtype
        else:
            dtypes[col] = dt

    # define the col
    col_dec_strs = {}

    for col in df_columns:
        if (check_dict[col] == "eq") or (check_dict[col] == "isin"):
            col_dec_strs[col] = (
                indents + f"{col}: {dtypes[col]} = pa.Field({check_strs[col]})"
            )
        else:
            col_dec_strs[col] = indents + f"{col}: {dtypes[col]} = pa.Field()"

    # define the config class

    # if 'basic_stats' is present then need to declare it here
    config_class_indent = indents * 2
    config_class_dec_str = indents + "class Config:"
    config_name_str = config_class_indent + f'name="{schema_name}"'
    config_strict_init = config_class_indent + f"strict=True"

    if gen_basic_stats:
        basic_stat_cols = basic_stats_dicts.keys()

        basic_stats_dict_varnames = {
            col: f"_{col}_basic_stats" for col in df_columns if col in basic_stat_cols
        }
        basic_stats_str_init = []
        for col in basic_stat_cols:
            col_item = f'"col":"{col}"'
            stat_item = f'"stats":{basic_stats_dicts[col]}'

            basic_stats_str_init.append(
                config_class_indent
                + f"{basic_stats_dict_varnames[col]}={{{col_item},{stat_item}}}"
            )

        basic_stats_str_init_block = "\n".join(basic_stats_str_init)

        check_stats_assign_str = [
            config_class_indent + f"check_stats = {basic_stats_dict_varnames[col]}"
            for col in basic_stat_cols
        ]
        check_stats_assign_str_block = "\n".join(check_stats_assign_str)

    # define full string
    if not inherit_from:
        inherit_str = "pa.DataFrameModel"
    else:
        inherit_str = inherit_from
    header_str = f"class {schema_name}({inherit_str}):"
    comment_str = f'{base_indent}"""\n{base_indent}An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.\n{base_indent}"""'

    col_str_block = "\n".join(col_dec_strs.values())

    definition_str = "\n".join(
        [
            header_str,
            comment_str,
            "",
            col_str_block,
            "",
            config_class_dec_str,
            "",
            config_name_str,
            config_strict_init,
            "",
        ]
    )

    if gen_basic_stats:
        definition_str = "\n".join(
            [
                definition_str,
                basic_stats_str_init_block,
                "",
                check_stats_assign_str_block,
            ]
        )

    return definition_str

def schema_tests(
    base_schema,
    dset_schema,
    base_schema_kwargs,
    dset_schema_kwargs,
    df,
    verbose: bool=False,
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
        
        print(
        base_schema_str    
        )
        raise ValueError("failed base schema test with error: " + str(e))
        
    try:
        dset_schema(df)
    except Exception as e:
    
        print("")
        
        print(
            dset_schema_str
        )
        raise ValueError(f"failed dataset schema test with error: {e}\n Printing schema for the input df..")
    
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


class OutPeakDF_Base(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    peak_idx: np.int64 = pa.Field()
    time_idx: pd.Int64Dtype = pa.Field()
    peak_prom: pd.Float64Dtype = pa.Field()
    whh: pd.Float64Dtype = pa.Field()
    whhh: pd.Float64Dtype = pa.Field()
    whh_left: pd.Float64Dtype = pa.Field()
    whh_right: pd.Float64Dtype = pa.Field()
    rel_height: pd.Float64Dtype = pa.Field()
    rl_width: pd.Float64Dtype = pa.Field()
    rl_wh: pd.Float64Dtype = pa.Field()
    rl_left: pd.Float64Dtype = pa.Field()
    rl_right: pd.Float64Dtype = pa.Field()

    class Config:
        name = "OutPeakDF_Base"
        strict = True


class OutPeakDFAssChrom(OutPeakDF_Base):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    peak_idx: np.int64 = pa.Field(eq=[0, 1, 2, 3])
    time_idx: pd.Int64Dtype = pa.Field(eq=[1507, 1899, 8000, 11000])
    peak_prom: pd.Float64Dtype = pa.Field(
        eq=[
            0.9994793466121925,
            0.07597734623882058,
            0.06225484797562456,
            0.9339044952972624,
        ]
    )
    whh: pd.Float64Dtype = pa.Field(
        eq=[
            270.5127402263288,
            172.53131767928176,
            706.3953656908734,
            235.48262868147685,
        ]
    )
    whhh: pd.Float64Dtype = pa.Field(
        eq=[
            21.340119296918846,
            18.334936769743333,
            1.3221146563326698,
            19.9395167271558,
        ]
    )
    whh_left: pd.Float64Dtype = pa.Field(
        eq=[
            1385.0966943344852,
            1809.5007097260184,
            7646.804206458902,
            10882.258685659264,
        ]
    )
    whh_right: pd.Float64Dtype = pa.Field(
        eq=[
            1655.609434560814,
            1982.0320274053001,
            8353.199572149775,
            11117.741314340741,
        ]
    )
    rel_height: pd.Float64Dtype = pa.Field(eq=[1.0, 1.0, 1.0, 1.0])
    rl_width: pd.Float64Dtype = pa.Field(
        eq=[9244.513493271112, 282.912258486617, 2921.8149969111128, 1876.0]
    )
    rl_wh: pd.Float64Dtype = pa.Field(
        eq=[
            -0.007597292892196739,
            16.71214900504366,
            -0.0075765027994752834,
            -0.007597292892196739,
        ]
    )
    rl_left: pd.Float64Dtype = pa.Field(eq=[687.4865067288889, 1736.0, 6455.0, 9932.0])
    rl_right: pd.Float64Dtype = pa.Field(
        eq=[9932.0, 2018.912258486617, 9376.814996911113, 11808.0]
    )

    class Config:
        name = "OutPeakDFAssChrom"
        strict = True


class OutSignalDF_Base(pa.DataFrameModel):
    time_idx: np.int64 = pa.Field(coerce=False)
    time: np.float64 = pa.Field(coerce=False)
    amp_raw: np.float64 = pa.Field(coerce=False)
    amp_corrected: Optional[np.float64] = pa.Field(coerce=False)
    amp_bg: Optional[np.float64] = pa.Field(coerce=False)
    amp_corrected_norm: Optional[np.float64] = pa.Field(coerce=False)
    amp_norm: Optional[np.float64] = pa.Field(coerce=False)


class OutSignalDF_ManyPeaks(OutSignalDF_Base):
    time_idx: np.int64 = pa.Field(ge=0, lt=7000)
    amp_raw: np.float64 = pa.Field(ge=6.425446e-48, le=3.989453e01)
    norm_amp: np.float64 = pa.Field(ge=0, le=1)


class OutSignalDF_AssChrom(OutSignalDF_Base):
    time_idx: np.int64 = pa.Field(
        coerce=False, in_range={"min_value": 0, "max_value": 149999}
    )
    time: np.float64 = pa.Field(
        coerce=False, in_range={"min_value": 0, "max_value": 149.99}
    )
    amp_raw: np.float64 = pa.Field(
        coerce=False,
        in_range={"min_value": -0.0298383947260937, "max_value": 42.69012166052291},
    )
    amp_corrected: Optional[np.float64] = pa.Field(
        coerce=False, in_range={"min_value": 0.007597293, "max_value": 42.703030473}
    )
    amp_bg: Optional[np.float64] = pa.Field(
        coerce=False,
        in_range={"min_value": -0.0075972928921949, "max_value": 0.0002465850460692323},
    )
    amp_corrected_norm: Optional[np.float64] = pa.Field(
        coerce=False, in_range={"min_value": 0, "max_value": 1}
    )
    amp_norm: Optional[np.float64] = pa.Field(
        coerce=False, in_range={"min_value": 0, "max_value": 1}
    )


class OutWindowDF_Base(pa.DataFrameModel):
    """
    Contains a recording of each window in the Chromatogram labeled with an ID and type
    with a time index corresponding to the time values in the time array.

    Spans the length of the chromatogram signal
    """

    time_idx: pd.Int64Dtype = pa.Field(coerce=False)
    window_idx: pd.Int64Dtype = pa.Field(coerce=False)
    window_type: str  # either 'peak' or 'np.int64erpeak'

    class Config:
        name = "BaseWindowDFSchema"
        strict = True


class OutWindowDF_ManyPeaks(OutWindowDF_Base):
    time_idx: pd.Int64Dtype = pa.Field(ge=636, le=5868)
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])

    class Config:
        name = "OutWindowDFManyPeaks"

class OutWindowDF_AssChrom(OutWindowDF_Base):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    window_idx: np.int64 = pa.Field()
    time_idx: pd.Int64Dtype = pa.Field()
    window_type: np.object_ = pa.Field(isin=['peak'])

    class Config:

        name="OutWindowDF_AssChrom"
        strict=True

        _window_idx_basic_stats={"col":"window_idx","stats":{'count': 11121.0, 'min': 1.0, 'max': 2.0, 'mean': 1.1686898660192429, 'std': 0.37449460083753666}}
        _time_idx_basic_stats={"col":"time_idx","stats":{'count': 11121.0, 'min': 687.0, 'max': 11807.0, 'mean': 6247.0, 'std': 3210.50050615165}}

        check_stats = _window_idx_basic_stats
        check_stats = _time_idx_basic_stats
        
        
class OutInitialGuessBase(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    window_idx: pd.Int64Dtype = pa.Field()
    peak_idx: pd.Int64Dtype = pa.Field()
    param: pd.CategoricalDtype = pa.Field()
    p0: pd.Float64Dtype = pa.Field()

    class Config:

        name="OutInitialGuessBase"
        strict=True

class OutInitialGuessAssChrom(OutInitialGuessBase):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    window_idx: pd.Int64Dtype = pa.Field()
    peak_idx: pd.Int64Dtype = pa.Field()
    param: pd.CategoricalDtype = pa.Field(isin=['amp', 'loc', 'whh', 'skew'])
    p0: pd.Float64Dtype = pa.Field(isin=[42.68783588672989, 15.07, 1.352563701131644, 0.0, 19.957724534443003, 18.99, 0.8626565883964088, 2.6518058154648143, 80.0, 3.531976828454367, 39.8866307472038, 110.0, 1.1774131434073842])

    class Config:

        name="OutInitialGuessAssChrom"
        strict=True

        _window_idx_basic_stats={"col":"window_idx","stats":{'count': 16.0, 'min': 1.0, 'max': 2.0, 'mean': 1.25, 'std': 0.4472135954999579}}
        _peak_idx_basic_stats={"col":"peak_idx","stats":{'count': 16.0, 'min': 0.0, 'max': 3.0, 'mean': 1.5, 'std': 1.1547005383792515}}

        check_stats = _window_idx_basic_stats
        check_stats = _peak_idx_basic_stats

class OutDefaultBoundsBase(pa.DataFrameModel):
    window_idx: pd.Int64Dtype
    peak_idx: pd.Int64Dtype
    param: pd.CategoricalDtype
    lb: pd.Float64Dtype = pa.Field(nullable=False)
    ub: pd.Float64Dtype = pa.Field(nullable=False)


class OutDefaultBoundsManyPeaks(OutDefaultBoundsBase):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    lb: pd.Float64Dtype = pa.Field(
        isin=[
            636.0,
            -np.inf,
            0.009999999999999998,
            3.9894525384046213,
            3.9894376712094743,
        ]
    )
    ub: pd.Float64Dtype = pa.Field(
        isin=[5868.0, np.inf, 2616.0, 398.9452538404621, 398.9437671209474]
    )


class OutDefaultBoundsAssChrom(OutDefaultBoundsBase):
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
            99.32,
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
            118.07,
            np.inf,
            9.375,
        ]
    )

    class Config:
        name = "OutDefaultBoundsAssChrom"
        strict = True


class OutWindowedSignalBase(OutSignalDF_Base):
    """
    The signal DF with the addition of a window ID column
    """

    window_idx: pd.Int64Dtype


class OutWindowedSignalManyPeaks(OutSignalDF_ManyPeaks):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])


class OutWindowedSignalAssChrom(OutWindowedSignalBase):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    time_idx: np.int64 = pa.Field()
    time: np.float64 = pa.Field()
    amp_raw: np.float64 = pa.Field()
    amp_corrected: np.float64 = pa.Field()
    amp_bg: np.float64 = pa.Field()
    window_idx: pd.Int64Dtype = pa.Field()

    class Config:
        name = "OutWindowedSignalAssChrom"
        strict = True

        _time_idx_basic_stats = {
            "col": "time_idx",
            "stats": {
                "count": 11121.0,
                "min": 687.0,
                "max": 11807.0,
                "mean": 6247.0,
                "std": 3210.50050615165,
            },
        }
        _time_basic_stats = {
            "col": "time",
            "stats": {
                "count": 11121.0,
                "min": 6.87,
                "max": 118.07,
                "mean": 62.46999999999999,
                "std": 32.1050050615165,
            },
        }
        _amp_raw_basic_stats = {
            "col": "amp_raw",
            "stats": {
                "count": 11121.0,
                "min": -0.0138380221313465,
                "max": 42.69012166052291,
                "mean": 2.8774412228164503,
                "std": 7.871276358953504,
            },
        }
        _amp_corrected_basic_stats = {
            "col": "amp_corrected",
            "stats": {
                "count": 11121.0,
                "min": -0.0138380221313465,
                "max": 42.68783588672989,
                "mean": 2.871086160845697,
                "std": 7.871945500463037,
            },
        }
        _amp_bg_basic_stats = {
            "col": "amp_bg",
            "stats": {
                "count": 11121.0,
                "min": 0.0,
                "max": 0.007843877938264132,
                "mean": 0.0063550619707533436,
                "std": 0.0019851337215562185,
            },
        }
        _window_idx_basic_stats = {
            "col": "window_idx",
            "stats": {
                "count": 11121.0,
                "min": 1.0,
                "max": 2.0,
                "mean": 1.1686898660192429,
                "std": 0.37449460083753666,
            },
        }

        check_stats = _time_idx_basic_stats
        check_stats = _time_basic_stats
        check_stats = _amp_raw_basic_stats
        check_stats = _amp_corrected_basic_stats
        check_stats = _amp_bg_basic_stats
        check_stats = _window_idx_basic_stats


class OutParamsBase(pa.DataFrameModel):
    window_idx: pd.Int64Dtype
    peak_idx: pd.Int64Dtype
    param: pd.CategoricalDtype
    p0: pd.Float64Dtype
    lb: pd.Float64Dtype
    ub: pd.Float64Dtype
    inbounds: bool

    class Config:
        strict = True


class OutParamManyPeaks(OutParamsBase):
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


class OutParamAssChrom(OutParamsBase):
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
            0.8626565883964088,
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
            99.32,
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
            118.07,
            9.375,
            np.inf,
        ]
    )
    inbounds: bool = pa.Field(
        eq=[
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
    )

    class Config:
        name = "OutParamAssChrom"
        strict = True

class OutPoptDF_Base(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    tbl_name: str = pa.Field()
    window_idx: pd.Int64Dtype = pa.Field()
    peak_idx: pd.Int64Dtype = pa.Field()
    amp: pd.Float64Dtype = pa.Field()
    loc: pd.Float64Dtype = pa.Field()
    whh: pd.Float64Dtype = pa.Field()
    skew: pd.Float64Dtype = pa.Field()

    class Config:

        name="OutPoptDF_Base"
        strict=True


class OutPoptDF_AssChrom(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    tbl_name: str = pa.Field(eq=['popt', 'popt', 'popt', 'popt'])
    window_idx: pd.Int64Dtype = pa.Field(eq=[1, 1, 1, 2])
    peak_idx: pd.Int64Dtype = pa.Field(eq=[0, 1, 2, 3])
    amp: pd.Float64Dtype = pa.Field(eq=[100.01095135095538, 99.95293327255338, 19.875942970961788, 99.95961060691138])
    loc: pd.Float64Dtype = pa.Field(eq=[14.907132807705182, 18.999888810333815, 79.95924682487205, 109.99318581350681])
    whh: pd.Float64Dtype = pa.Field(eq=[1.0043771966152746, 1.9992972294398816, 2.987924864844685, 0.999753924385777])
    skew: pd.Float64Dtype = pa.Field(eq=[0.11692923230492229, 0.00020749742748570234, 0.017105776889882674, 0.008542771024883706])

    class Config:

        name="OutPoptDF_AssChrom"
        strict=True

def isArrayLike(x: Any):
    if not any(x):
        raise ValueError("x is None")

    if not hasattr(x, "__array__"):
        return False
    else:
        return True


class OutReconDFBase(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    peak_idx: pd.Int64Dtype = pa.Field()
    time: np.float64 = pa.Field()
    unmixed_amp: np.float64 = pa.Field()

    class Config:

        name="OutReconDFBase"
        strict=True


class OutReconDF_AssChrom(OutReconDFBase):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    peak_idx: pd.Int64Dtype = pa.Field()
    time: np.float64 = pa.Field()
    unmixed_amp: np.float64 = pa.Field()

    class Config:

        name="OutReconDF_AssChrom"
        strict=True

        _peak_idx_basic_stats={"col":"peak_idx","stats":{'count': 60000.0, 'min': 0.0, 'max': 3.0, 'mean': 1.5, 'std': 1.1180433058162647}}
        _time_basic_stats={"col":"time","stats":{'count': 60000.0, 'min': 0.0, 'max': 149.99, 'mean': 74.995, 'std': 43.30163094142494}}
        _unmixed_amp_basic_stats={"col":"unmixed_amp","stats":{'count': 60000.0, 'min': 0.0, 'max': 39.896479288479064, 'mean': 0.5329990636689699, 'std': 3.395449575437415}}

        check_stats = _peak_idx_basic_stats
        check_stats = _time_basic_stats
        check_stats = _unmixed_amp_basic_stats

class OutPeakReportBase(OutPoptDF_Base):
    
    tbl_name: pd.StringDtype = pa.Field(eq="peak_report")
    retention_time: pd.Float64Dtype
    unmixed_area: pd.Float64Dtype
    unmixed_maxima: pd.Float64Dtype


class OutPeakReportAssChrom(OutPeakReportBase):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    tbl_name: str = pa.Field(eq=['peak_report', 'peak_report', 'peak_report', 'peak_report'])
    window_idx: pd.Int64Dtype = pa.Field(eq=[1, 1, 1, 2])
    peak_idx: pd.Int64Dtype = pa.Field(eq=[0, 1, 2, 3])
    retention_time: pd.Float64Dtype = pa.Field(eq=[0.1490713280770518, 0.18999888810333815, 0.7995924682487205, 1.099931858135068])
    loc: pd.Float64Dtype = pa.Field(eq=[14.907132807705182, 18.999888810333815, 79.95924682487205, 109.99318581350681])
    amp: pd.Float64Dtype = pa.Field(eq=[100.01095135095538, 99.95293327255338, 19.875942970961788, 99.95961060691138])
    whh: pd.Float64Dtype = pa.Field(eq=[1.0043771966152746, 1.9992972294398816, 2.987924864844685, 0.999753924385777])
    skew: pd.Float64Dtype = pa.Field(eq=[0.11692923230492229, 0.00020749742748570234, 0.017105776889882674, 0.008542771024883706])
    unmixed_area: np.float64 = pa.Field(eq=[10001.095135095537, 9995.293327255338, 1987.5942970961787, 9995.96106069114])
    unmixed_maxima: np.float64 = pa.Field(eq=[39.896479288479064, 19.94473400496891, 2.654046807908235, 39.8888570115766])

    class Config:

        name="OutPeakReportAssChrom"
        strict=True

