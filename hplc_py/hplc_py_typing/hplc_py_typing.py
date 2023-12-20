from typing import Any

import pandas as pd
import pandera as pa
import pandera.typing as pt
import pandera.extensions as extensions

import numpy.typing as npt
import numpy as np
from typing import Optional

@extensions.register_check_method(statistics=["col","stats"]) #type: ignore
def check_stats(df, *, col: str, stats:dict):
    '''
    Test basic statistics of a dataframe. Ideal for validating data in large frames.
    
    Provide stats as a dict of {'statistic_name' : expected_val}.
    Currently tested for: count, mean, std, min, max
    '''
    # statistics that I have checked to behave as expected
    valid_stats = ['count','mean','std','min','max']
    
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
        
        col_stats[stat]=col_stat
    
    # check all results, generating a report string for failures only then raising a Value Error
    error_strs = []
    if not all(checks.values()):
        for stat, test in checks.items():
            if not test:
                error_str = f"{col} has failed {stat} check. Expected {stat} is {stats[stat]}, but {col} {stat} is {col_stats[stat]}"
                error_strs.append(error_str)
        raise ValueError("\n"+"\n".join(error_strs))
    else:
        # if all checks pass, move on
        return True

def interpret_model(df,
                    schema_name=None,
                    inherit_from: str="",
                    check_dict:dict={},
                    ):
    '''
    Output a string representation of a dataframe schema DataFrameModel with datatypes and checks.
    Outputs both a base model and a specific model.
    
    check_dict: check type for each column. specify column name as key, check type: 'eq', 'isin', 'basic_stats'
    
    '''
    custom_indent_mag = 1
    indent_str = ' '
    base_indent_mag = 4
    base_indent = "".join([indent_str]*base_indent_mag)
    indents = "".join(base_indent*custom_indent_mag)
    
    df_columns = df.columns
    
    def eq_checkstr(series):
        return f"eq={series.tolist()}"
    
    def isin_checkstr(series):
        return f"isin={series.unique().tolist()}"
    
    if not check_dict:
        raise ValueError(f"Please provide a check_dict of columns:{df_columns}")
    
    # generate the check strings
    
    check_strs = {}
    if 'basic_stats' in check_dict.values():
        basic_stats = ['count','min','max','mean','std']
        basic_stats_dicts = {}
    
    # prepare the checks
    
    for col, check_str in check_dict.items():
        series = df[col]
        if check_str=='eq':
            check_strs[col]=eq_checkstr(series)
        elif check_str=='isin':
            check_strs[col]=isin_checkstr(series)
        elif check_str=='basic_stats':
            # must be a numerical column
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"{col} must be numeric to use 'basic_stats' option")
            else:
                basic_stats_dicts[col]=dict(zip(basic_stats, series.describe()[basic_stats].to_list()))

    
    
    # define datatypes with appending/preppending if necessary for imports
    dtype={}
    for col in df_columns:
        
        dt=str(df[col].dtype)

        amended_dtype=None
        
        if any(str.isupper(c) for c in dt):
            amended_dtype='pd.'+ dt +"Dtype()"
        elif any(str.isnumeric(c) for c in dt):
            amended_dtype="np." +dt
        elif dt=='object':
            amended_dtype="np.object_"
        
        if amended_dtype:
            dtype[col]=amended_dtype
        else:
            dtype[col]=dtype
    
    
    # define the col
    col_dec_strs = {}
    
    for col in df_columns:
        if (check_dict[col]=='eq') or (check_dict[col]=='isin'):
            col_dec_strs[col]=indents+f"{col}: {dtype[col]} = pa.Field({check_strs[col]})"
        else:
            col_dec_strs[col]=indents+f"{col}: {dtype[col]} = pa.Field()"

    # define the config class
    
    # if 'basic_stats' is present then need to declare it here
    config_class_indent = indents*2
    config_class_dec_str = indents+"class Config:"
    config_name_str = config_class_indent+f"name=\"{schema_name}\""
    
    basic_stat_cols = basic_stats_dicts.keys()
    
    basic_stats_dict_varnames = {col: f"_{col}_basic_stats" for col in df_columns if col in basic_stat_cols}
    basic_stats_str_init = []
    for col in basic_stat_cols:
        
        col_item = f"\"col\":\"{col}\""
        stat_item = f"\"stats\":{basic_stats_dicts[col]}"
        
        basic_stats_str_init.append(config_class_indent+f"{basic_stats_dict_varnames[col]}={{{col_item},{stat_item}}}")
        
    basic_stats_str_init_block = "\n".join(basic_stats_str_init)
    
    check_stats_assign_str = [config_class_indent+f"check_stats = {basic_stats_dict_varnames[col]}" for col in basic_stat_cols]
    check_stats_assign_str_block = "\n".join(check_stats_assign_str)
    
    # define full string    
    if not inherit_from:
        inherit_str = "pa.DataFrameModel"
    else:
        inherit_str = inherit_from
    header_str = f"class {schema_name}({inherit_from}):"
    comment_str =f"{base_indent}\"\"\"\n{base_indent}An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.\n{base_indent}\"\"\""
    
    col_str_block = "\n".join(col_dec_strs.values())
    
    definition_str = "\n".join([
        header_str,
        comment_str,
        "",
        col_str_block,
        "",
        config_class_dec_str,
        "",
        config_name_str,
        "",
        basic_stats_str_init_block,
        "",
        check_stats_assign_str_block,
        
    ])
    print("")
    print(definition_str)
    return None
    

class SignalDFIn(pa.DataFrameModel):
    """
    The base signal, with time and amplitude directions
    """

    tbl_name: Optional[str] = pa.Field(eq="testsignal")
    time: np.float64
    amp_raw: np.float64
    
class SignalDFInAssChrom(SignalDFIn):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    tbl_name: np.object_ = pa.Field(isin=['testsignal'])
    time: np.float64 = pa.Field()
    amp_raw: np.float64 = pa.Field()

    class Config:

        name="SignalDFInAssChrom"

        _time_basic_stats={"col":"time","stats":{'count': 15000.0, 'min': 0.0, 'max': 149.99, 'mean': 74.995, 'std': 43.30271354083945}}
        _amp_raw_basic_stats={"col":"amp_raw","stats":{'count': 15000.0, 'min': -0.0298383947260937, 'max': 42.69012166052291, 'mean': 2.1332819642622307, 'std': 6.893591961394714}}

        check_stats = _time_basic_stats
        check_stats = _amp_raw_basic_stats

class OutPeakDF_Base(pa.DataFrameModel):
    """
    Containsnp.information about each detected peak, used for profiling
    """

    peak_idx: pd.Int64Dtype = pa.Field()
    time_idx: pd.Int64Dtype = pa.Field(
        coerce=False
    )  # the time idx values corresponding to the peak maxima location
    peak_prom: pd.Float64Dtype = pa.Field(coerce=False)
    whh: pd.Float64Dtype = pa.Field(coerce=False)
    whhh: pd.Float64Dtype = pa.Field(coerce=False)
    whh_left: pd.Float64Dtype = pa.Field(coerce=False)
    whh_right: pd.Float64Dtype = pa.Field(coerce=False)
    rl_width: pd.Float64Dtype = pa.Field(coerce=False)
    rl_wh: pd.Float64Dtype = pa.Field(coerce=False)
    rl_left: pd.Float64Dtype = pa.Field(coerce=False)
    rl_right: pd.Float64Dtype = pa.Field(coerce=False)

    class Config:
        name = "OutPeakDF"
        strict = False

    @pa.dataframe_check
    def check_null(cls, df: pd.DataFrame) -> bool:
        return df.shape[0] > 0


class OutPeakDF_ManyPeaks(OutPeakDF_Base):
    """
    commented out fields are specific to 'test_many_peaks.csv'. Until I find a method of parametrizing
    """

    time_idx: pd.Int64Dtype = pa.Field(
        isin=[1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
    )
    peak_prom: pd.Float64Dtype = pa.Field(
        isin=[
            0.9221250775321379,
            0.9129314313007107,
            0.9131520182120347,
            0.9135276076962014,
            0.9140168870790715,
            0.9138066553780264,
            0.9133958162236926,
            0.9131460523698959,
            0.9130556194514102,
            1.0,
        ],
    )

    whh: pd.Float64Dtype = pa.Field(isin=[220.0, 221.0, 235.0])
    whhh: pd.Float64Dtype = pa.Field(isin=[0])
    whh_left: pd.Float64Dtype = pa.Field(
        isin=[888, 1389, 1889, 2389, 2889, 3389, 3889, 4389, 4889, 5382]
    )
    whh_right: pd.Float64Dtype = pa.Field(
        isin=[1110, 1610, 2110, 2610, 3110, 3610, 4110, 4610, 5110, 5617]
    )

    rl_width: pd.Float64Dtype = pa.Field(
        isin=[489.0, 490.0, 2475.0, 492.0, 496.0, 494.0, 491.0, 5233.0]
    )
    rl_wh: pd.Float64Dtype = pa.Field(isin=[0])
    rl_left: pd.Float64Dtype = pa.Field(
        isin=[774, 1250, 1750, 2250, 2750, 3255, 3758, 4259, 4760, 636]
    )
    rl_right: pd.Float64Dtype = pa.Field(
        isin=[3249, 1739, 2240, 2742, 3246, 3749, 4250, 4749, 5250, 5869]
    )


class OutPeakDF_AssChrom(OutPeakDF_Base):
    """
    commented out fields are specific to 'test_many_peaks.csv'. Until I find a method of parametrizing
    """

    peak_idx: pd.Int64Dtype = pa.Field(eq=[0, 1, 2, 3])
    time_idx: pd.Int64Dtype = pa.Field(eq=[1507, 1899, 8000, 11000])
    peak_prom: pd.Float64Dtype = pa.Field(
        nullable=True,
        eq=[1.0, 0.0760169246981745, 0.06228727805122132, 0.9343909891207713],
    )
    peak_prom: pd.Float64Dtype = pa.Field(
        nullable=True,
        eq=[1.0, 0.0760169246981745, 0.06228727805122132, 0.9343909891207713],
    )

    whh: pd.Float64Dtype = pa.Field(
        eq=[
            270.5127402224607,
            172.53131767824425,
            706.3953657358797,
            235.48262868254096,
        ]
    )
    whhh: pd.Float64Dtype = pa.Field(
        eq=[0.5, 0.4296134901634462, 0.031144125962935136, 0.46719549456038567]
    )
    whh_left: pd.Float64Dtype = pa.Field(
        eq=[1385.096694334869, 1809.5007097229984, 7646.804206486151, 10882.25868565873]
    )
    whh_right: pd.Float64Dtype = pa.Field(
        eq=[1655.6094345573297, 1982.0320274012427, 8353.19957222203, 11117.74131434127]
    )
    rel_height: pd.Float64Dtype = pa.Field(eq=[1.0, 1.0, 1.0, 1.0])
    rl_width: pd.Float64Dtype = pa.Field(
        eq=[8721.0, 282.91225849251373, 2921.8148514764607, 1409.0]
    )
    rl_wh: pd.Float64Dtype = pa.Field(
        eq=[0.0, 0.3916050278143589, 4.869373244759112e-07, 0.0]
    )
    rl_left: pd.Float64Dtype = pa.Field(eq=[687.0, 1736.0, 6455.000000008679, 10297.0])
    rl_right: pd.Float64Dtype = pa.Field(
        isin=[9408.0, 2018.9122584925137, 9376.81485148514, 11706.0]
    )


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
        coerce=False, in_range={'min_value':-0.0298383947260937, 'max_value':42.69012166052291}
    )
    amp_corrected: Optional[np.float64] = pa.Field(
        coerce=False, in_range= {'min_value':0.007597293, 'max_value':42.703030473}
    )
    amp_bg: Optional[np.float64] = pa.Field(
        coerce=False, in_range={'min_value':-0.0075972928921949, 'max_value':0.0002465850460692323}
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
        name='BaseWindowDFSchema'
        strict=True



class OutWindowDF_ManyPeaks(OutWindowDF_Base):
    time_idx: pd.Int64Dtype = pa.Field(ge=636, le=5868)
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])

    class Config:
        name='OutWindowDFManyPeaks'

class OutWindowDF_AssChrom(OutWindowDF_Base):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1, 2])
    time_idx: pd.Int64Dtype = pa.Field(ge=687, le=11705)
    
    class Config:
        name='OutWindowDFAssChrom'


class OutInitialGuessBase(pa.DataFrameModel):
    """
    The DF containing the initial guesses of each peak by window, created by `PeakDeconvolver.p0_factory`
    """

    window_idx: pd.Int64Dtype
    peak_idx: pd.Int64Dtype
    param: pd.CategoricalDtype
    p0: pd.Float64Dtype


class OutInitialGuessManyPeaks(OutInitialGuessBase):
    """
    The DF containing the initial guesses of each peak by window, created by `PeakDeconvolver.p0_factory`
    """

    pass
    window_idx: pd.Int64Dtype = pa.Field(
        eq=[
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )
    peak_idx: pd.Int64Dtype = pa.Field(
        eq=[
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            9,
            9,
            9,
            9,
        ]
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
            39.89437671209474,
            1000.0,
            110.99499155692575,
            0.0,
            39.89452538404621,
            1500.0,
            110.27465597802689,
            0.0,
            39.89452538404621,
            2000.0,
            110.30883814845754,
            0.0,
            39.89452538404621,
            2500.0,
            110.36057294811258,
            0.0,
            39.89452538404621,
            3000.0,
            110.41766297678032,
            0.0,
            39.89452538404621,
            3500.0,
            110.40327814820625,
            0.0,
            39.89452538404621,
            4000.0,
            110.35506235294383,
            0.0,
            39.89452538404621,
            4500.0,
            110.30903237451776,
            0.0,
            39.89452538404621,
            5000.0,
            110.28355710004189,
            0.0,
            39.89437671209474,
            5500.0,
            117.53195594948511,
            0.0,
        ]
    )


class OutInitialGuessAssChrom(OutInitialGuessBase):
    """
    The DF containing the initial guesses of each peak by window, created by `PeakDeconvolver.p0_factory`
    """

    window_idx: pd.Int64Dtype = pa.Field(isin=[1, 2])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    p0: pd.Float64Dtype = pa.Field(
        eq=[
            42.69012166052291,
            1507.0,
            135.25637011123035,
            0.0,
            19.96079318035132,
            1899.0,
            86.26565883912212,
            0.0,
            2.659615202676218,
            8000.0,
            353.1976828679399,
            0.0,
            39.89422804014327,
            11000.0,
            117.74131434127048,
            0.0,
        ]
    )


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
            4.269012166052291,
            687.0,
            -np.inf,
            1.0,
            1.9960793180351322,
            687.0,
            -np.inf,
            1.0,
            0.2659615202676218,
            687.0,
            -np.inf,
            1.0,
            3.989422804014327,
            10297.0,
            -np.inf,
            1.0,
        ]
    )
    ub: pd.Float64Dtype = pa.Field(
        eq=[
            426.90121660522914,
            9407.0,
            np.inf,
            4360.0,
            199.6079318035132,
            9407.0,
            np.inf,
            4360.0,
            26.59615202676218,
            9407.0,
            np.inf,
            4360.0,
            398.9422804014327,
            11705.0,
            np.inf,
            704.0,
        ]
    )


class OutWindowedSignalBase(OutSignalDF_Base):
    """
    The signal DF with the addition of a window ID column
    """

    window_idx: pd.Int64Dtype


class OutWindowedSignalManyPeaks(OutSignalDF_ManyPeaks):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])


class OutWindowedSignalAssChrom(OutSignalDF_AssChrom):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1, 2])


class OutParamsBase(pa.DataFrameModel):
    window_idx: pd.Int64Dtype
    peak_idx: pd.Int64Dtype
    param: pd.CategoricalDtype
    p0: pd.Float64Dtype
    lb: pd.Float64Dtype
    ub: pd.Float64Dtype
    inbounds: bool

    class Config:
        strict=True

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
    window_idx: pd.Int64Dtype = pa.Field(isin=[1, 2])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    p0: pd.Float64Dtype = pa.Field(
        eq=[
            42.69012166052291,
            1507.0,
            135.25637011123035,
            0.0,
            19.96079318035132,
            1899.0,
            86.26565883912212,
            0.0,
            2.659615202676218,
            8000.0,
            353.1976828679399,
            0.0,
            39.89422804014327,
            11000.0,
            117.74131434127048,
            0.0,
        ]
    )
    lb: pd.Float64Dtype = pa.Field(
        eq=[
            4.269012166052291,
            687.0,
            1.0,
            -np.inf,
            1.9960793180351322,
            687.0,
            1.0,
            -np.inf,
            0.2659615202676218,
            687.0,
            1.0,
            -np.inf,
            3.989422804014327,
            10297.0,
            1.0,
            -np.inf,
        ]
    )
    ub: pd.Float64Dtype = pa.Field(
        eq=[
            426.90121660522914,
            9407.0,
            4360.0,
            np.inf,
            199.6079318035132,
            9407.0,
            4360.0,
            np.inf,
            26.59615202676218,
            9407.0,
            4360.0,
            np.inf,
            398.9422804014327,
            11705.0,
            704.0,
            np.inf,
        ]
    )
    inbounds: bool = pa.Field(isin=[True])


class OutPoptBase(pa.DataFrameModel):
    pass


class OutPoptManyPeaks(OutPoptBase):
    pass


class OutPoptAssChrom(OutPoptBase):
    pass


def isArrayLike(x: Any):
    if not any(x):
        raise ValueError("x is None")

    if not hasattr(x, "__array__"):
        return False
    else:
        return True


class OutReconDFBase(pa.DataFrameModel):
    """
    it is always better to have a set number of columns and very the length than it is
    to have a varying number of columns. Thus it is best to store ReconDF in long form
    then pivot where necessary. Of course the most space efficient form will be one that
    does not repeat the times. That in itself is a side-study that we do not have time for.
    """

    peak_idx: pd.Int64Dtype
    time_idx: pd.Int64Dtype
    unmixed_amp: pd.Float64Dtype


class OutReconDFManyPeaks(OutReconDFBase):
    pass


class OutReconDFAssChrom(OutReconDFBase):
    pass


class OutPeakReportBase(OutPoptBase):
    unmixed_area: pd.Float64Dtype
    unmixed_maxima: pd.Float64Dtype
    tbl_name: pd.StringDtype = pa.Field(eq="peak_report")


class OutPeakReportManyPeaks(OutPeakReportBase):
    pass


class OutPeakReportAssChrom(OutPeakReportBase):
    pass
