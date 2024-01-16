import pandas as pd


def interpret_model(
    df,
    schema_name: str = "",
    inherit_from: str = "",
    is_base: bool = False,
    check_dict: dict = {},
    pandas_dtypes: bool = True,
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
        if len(df) <= 10:
            check_dict = {col: "eq" for col in df}
        else:
            check_dict = {
                col: "basic_stats" if pd.api.types.is_numeric_dtype(df[col]) else "isin"
                for col in df
            }

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
        elif dt == "string":
            amended_dtype = "pd.StringDtype"
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