import pandas as pd
import pyperclip

class Modelinterpreter:
    
    def __init__(
        self,
        custom_indent_mag = 1,
        indent_str = " ",
        base_indent_mag = 4,
    ):
        self.base_indent = "".join([indent_str] * base_indent_mag),
        self.indents = "".join(self.base_indent * custom_indent_mag)
    
    def gen_eq_checkstr(self, series):
        return f"eq={series.tolist()}"

    def gen_isin_checkstr(self, series):
        return f"isin={series.unique().tolist()}"

    
    def interpret_model(
        self,
        df: pd.DataFrame,
        schema_name: str = "",
        inherit_from: str = "",
        is_base: bool = False,
        check_dict: dict = {},
        pandas_dtypes: bool = True,
        clipboard: bool = False,
    ):
        df_columns = df.columns
        """
        Output a string representation of a dataframe schema DataFrameModel with datatypes and checks.
        Outputs both a base model and a specific model.

        :param check_dict: check type for each column. specify column name as key, check type: 'eq', 'isin', 'basic_stats'. if no checkdict is passed and is_base is false, generate an opinionated check dict based on types and dataframe size. use 'eq' if frame is less than 10 rows long, else use 'basic_stats' and 'isin'. Problem with this compression approach is that you lose information on ordering.
        :param clipboard: Whether to copy the schema definition string to the clipboard
        :type clipboard: Bool
        """

        gen_basic_stats = "basic_stats" in check_dict.values()
        
        breakpoint()
        
        # assert no check_dict passed if is_base is True

        if (is_base) and (check_dict):
            raise ValueError("do not provide a check_dict if is_base == True")


        if is_base:
            check_dict = {col: "" for col in df_columns}

        # generate opinionated default check_dict

        if not check_dict and not is_base:
            
            check_dict = self.apply_defaults(df)

        # generate the check strings
        
        check_strs, basic_stats_dicts = self.gen_checkstrs(df=df, check_dict=check_dict, gen_basic_stats=gen_basic_stats)
        breakpoint()
        df_columns = df.columns.tolist()
        # define datatypes with appending/preppending if necessary for imports
        dtypes = self.substitute_dtype_substrs(df, df_columns)

        # generate the column field declarations
        col_dec_strs = self.gen_col_fields(check_dict, self.indents, df_columns, check_strs, dtypes)

        # define the config class

        config_class_dec_str, config_name_str, config_strict_init, basic_stats_str_init_block, check_stats_assign_str_block = self.gen_config_str(schema_name, self.indents, df_columns, gen_basic_stats=gen_basic_stats)

        # define full string
        
        header_str = self.gen_class_sig_str(schema_name, inherit_from)
        comment_str = self.gen_docstring_str(self.base_indent)

        schema_str = self.gen_sch_str(gen_basic_stats, col_dec_strs, config_class_dec_str, config_name_str, config_strict_init, basic_stats_str_init_block, check_stats_assign_str_block, header_str, comment_str)
        
        if clipboard:
            pyperclip.copy(schema_str)

        return schema_str

    def gen_sch_str(self, gen_basic_stats, col_dec_strs, config_class_dec_str, config_name_str, config_strict_init, basic_stats_str_init_block, check_stats_assign_str_block, header_str, comment_str):
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
        
        definition_str +="\n"
        return definition_str

    def gen_docstring_str(self, base_indent):
        comment_str = f'{base_indent}"""\n{base_indent}An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.\n{base_indent}"""'
        return comment_str

    def gen_class_sig_str(self, schema_name, inherit_from):
        if not inherit_from:
            inherit_str = "pa.DataFrameModel"
        else:
            inherit_str = inherit_from
        header_str = f"\nclass {schema_name}({inherit_str}):"
        return header_str

    def gen_config_str(self, schema_name, indents, df_columns, gen_basic_stats: bool=False):
        """
        Generate the Config definition
        """
        
        config_class_indent = indents * 2
        config_class_dec_str = indents + "class Config:"
        config_name_str = config_class_indent + f'name="{schema_name}"'
        config_strict_init = config_class_indent + f"strict=True"

        # if 'basic_stats' is present then need to declare it here
        
        breakpoint()
        if gen_basic_stats:
            basic_stat_cols = self.basic_stats_dicts.keys()

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
            
        return config_class_dec_str,config_name_str,config_strict_init,basic_stats_str_init_block,check_stats_assign_str_block

    def gen_col_fields(self, check_dict, indents, df_columns: list[str], check_strs, dtypes):
        
        col_dec_strs = {}
        
        for col in df_columns:
            if (check_dict[col] == "eq") or (check_dict[col] == "isin"):
                col_dec_strs[col] = (
                    indents + f"{col}: {dtypes[col]} = pa.Field({check_strs[col]})"
                )
            else:
                col_dec_strs[col] = indents + f"{col}: {dtypes[col]} = pa.Field()"
        return col_dec_strs

    def substitute_dtype_substrs(self, df: pd.DataFrame, df_columns: list[str]):
        """
        Pandas datatype string represntations are not the same as their Python class name (annoyingly). This makes it necesssary find and replace where necessary.
        """

        dtypes = {}
        for col in df_columns:
            dtype = str(df[col].dtype)

            amended_dtype = None
            if dtype == "object":
                amended_dtype = "object_"
            elif dtype == "category":
                amended_dtype = "CategoricalDtype"
            elif dtype == "string":
                amended_dtype = "StringDtype"
            if amended_dtype:
                dtypes[col] = amended_dtype
            else:
                dtypes[col] = dtype
        return dtypes

    def gen_checkstrs(self, df: pd.DataFrame, check_dict: dict, gen_basic_stats: bool):
        """
        For 'eq' or 'isin' we can take advantage of the built-in checks and define them 
        in the Field definition, but for 'basic_stats' we need to pass the check values
        to the custom check assignment in the Config subclass. Thus this function iterates
        through the predefined (or default) check dict values of each column and assigns
        the column to the relevant check
        """
        basic_stats_checks = {}
        basic_stats = []

        basic_stats = ["count", "min", "max", "mean", "std"]
        
        check_types_values = {}
        
        for col, check_type in check_dict.items():
            series = df[col]
            
            if check_type == "eq":
                check_types_values[col] = self.gen_eq_checkstr(series)
            elif check_type == "isin":
                check_types_values[col] = self.gen_isin_checkstr(series)
            elif check_type == "basic_stats":
                # must be a numerical column
                if not pd.api.types.is_numeric_dtype(series):
                    raise ValueError(f"{col} must be numeric to use 'basic_stats' option")
                
                # forms a dict whose keys are the statistic in question and the values are the statistic value
                basic_stats_checks[col] = dict(
                    zip(basic_stats_checks, series.describe()[basic_stats].to_list())
                )
                breakpoint()
        return check_types_values, basic_stats_checks

    def apply_defaults(self, df):

        if len(df) <= 10:
            check_dict = {col: "eq" for col in df}
        else:
            check_dict = {
                    col: "basic_stats" if pd.api.types.is_numeric_dtype(df[col]) else "isin"
                    for col in df
                }
            
        return check_dict