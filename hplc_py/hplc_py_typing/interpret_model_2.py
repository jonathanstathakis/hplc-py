"""
Mk 2 of interpret model. The first one was far too spaghetti and it will be easier to simply start from scratch
"""
from typing import Type, Self
import pandas as pd
import numpy as np
import pandera as pa
from pandera.typing import Series, DataFrame
from dataclasses import dataclass, field

class InterpetModel:
    """
    InterpetModel will take a frame and parameter input and produce a DataFrame model schema string with a number of checks, one for each column of the input frame.
    
    The process will:
    
    1. read the frame and make decisions about the number of column fields, the name of the field, the datatype of the field.
    2. First the class signature - the name of the class, anything it inherits from.
    3. based on information garnered in 1. it will construct checks to either be inserted in the column Field (if using built-in checks) or delayed for the 'basic_stats' check.
    3. Based on parameter input, the Config will be assembled. Options will include strictness, ordering, the name attribute.
    4. finally the basic_stats checks will be defined. This requires adding the call to `basic_stats` for each column, with the specified `basic_stats` parameter input.
    5. Once each of these strings are generated, we will assemble collect them in an arranger function which will produce the complete string for output.
    """
    def __init__(self):
        pass
    
    def gen_schema_def(
        self,
        df: pd.DataFrame,
        sch_name: str="PleaseNameMe",
        inherit_from: str="DataFrameModel",
        checks: dict = {},
        
    ):
        if not checks:
            checks = self._gen_default_checks(df)
        
        signature = self._gen_class_signature(class_name=sch_name, inherit_from=inherit_from)
        
        docstring = self._gen_docstring()
        
        colfields = self._gen_column_fields(df, checks)
        
    def _gen_column_fields(
        self,
        df: pd.DataFrame,
        checks: dict,
    )->str:
        """
        The column fields consist of their name, their datatype, and Field parameter 
        values, for example any built-in checks. Currently InterpretModel takes 
        advantage of the 'eq' for small frames, and 'isin' for columns of longer 
        non-numeric frames. Thus we need to accept both the frame and pre-decided 
        checks. But regardless of whether a particular column is checked here or in the 
        custom check section, it needs to be defined here as well. Use four functions 
        here - one for the field names, one for teh datatypes, one for field parameters,
        and one for the built-in checks. Use a frame to represent the fields, as they 
        are linear structures of somewhat equal length
        """
        class FieldBlock:
            def __init__(self, df: pd.DataFrame, checks: dict):
                self.df = df
                self.df_cols = df.columns
                self.df_dtypes = df.dtypes
                self.checks = checks
                
            def gen_fields(
                self
            ):  
            
                fields = [Field(self.df[col], self.checks[col]) for col in self.df]
                fields = [field.gen_field() for field in fields]
                
                breakpoint()

            
        cf = FieldBlock(df, checks)
        
        cf.gen_fields()
        
    def _gen_docstring(self)->str:
        """
        Not currently fleshed out, not sure if the schema should have a docstring considereing there is a 'description' metadata field in the Config inner class.
        """
        docstring = "\"\"\"\nThis is a automatically generated schema.\n\"\"\""
        return docstring
            
    def _gen_default_checks(self, df: pd.DataFrame)->str:
        """
        Generate default checks for the input df if no checks dict is passed. This is opinionated - in the interest of reducing the size of the schema string, any frame longer than 10 rows will be checked either against the 'basic statistics' of its numeric columns or unique entries in the case of non-numerics.
        
        The 'basic statistcs' are inspired by the default statistics in `pd.describe()` - the 'count', 'min', 'max', 'mean', and 'std'. This will be a pseudo-hashkey, a unique identifier for the frame, as each value is stored as a high decimal place float. Any subtle changes to the frame should modify the statistics enough to fail validation.
        """
        checks = {}
        if len(df) <= 10:
            for col in df:
                checks[col]='eq'
        else:
            for col in df:
                if pd.api.types.is_numeric_dtype(df[col]):
                    checks[col]='basic_stats'
                else:
                    checks[col]='isin'
                    
        return checks
    
    def _gen_class_signature(
        self,
        class_name: str,
        inherit_from: str,
    )->str:
        """
        Generate the string for the class signature with two variables - the name of the class and any inheritance. By default it inherits from pandera DataFrameModel - a requirement. Alternatively it can inherit from a superclass, but that class must inherit at some point from DataFrameModel.
        
        :param class_name: The name of the DataFrameModel class to be generated.
        :type class_name: str
        :inherit_from: a string identifying any classes to inherit from. This will typically be left to the default DataFrameModel.
        :type inherit_from: str
        """
        
        signature = f"\nclass {class_name}({inherit_from}):\n"
        
        return signature

class Field:
    """
    The class handling each column field, the column values, dtype and 
    check
    """
    def __init__(self, series: pd.Series, check: str):
        self.name = str(series.name)
        self._series = series
        self.check = check
        
        self.dtype_replacements = {
            "numpy.dtypes.Int64DType":"np.int64",
            "numpy.dtypes.ObjectDType":"object_",
            "category": "CategoricalDtype",
            "string": "StringDtype",
        }
        
    def gen_field(
        self
    )->Self:
        
        self.checkstr = self.get_builtin_checkstrs(self._series)
        self.dtype = self.get_field_dtype()
        
        return self
        
    def get_field_dtype(
        self
    ):
        """
        Get the datatype annotation for the field. Pandas is annoying and their string
        representation of their datatypes is not the actual code definition, thus parsing
        is necessary.
        
        - use 'type' attribute
        """
        
        series_dtype = self._series.dtype
        import ipdb; ipdb.set_trace()
        
        # self.dtype = self.dtype_replacements[series_dtype]
        
        return self
        
    
    def get_builtin_checkstrs(
        self,
        series: pd.Series
    ):
        """
        Use while iterating over the fields frame to append checkstrings 
        depending on the value of 'checks' then join back to the fields 
        frame.
        """
        checkstr = ""
        if self.check=='eq':
            checkstr=self.gen_eq_checkstr(series)
        elif self.check=='isin':
            checkstr=self.gen_isin_checkstr(series)
        else:
            checkstr=""
        
        return checkstr
        
    def gen_eq_checkstr(
        self,
        series: pd.Series,
    ):
        """
        For a given series, produce a string to call the 'eq' check, 
        containing the values of the column in order.
        """
        
        eq_str = "eq="+str(series.to_list())
        
        return eq_str
    
    def gen_isin_checkstr(
        self,
        series: pd.Series,
    ):
        """
        For a given series, produce a string to call an 'isin' check, containing unique values of the column
        """
        
        isin_str = "isin="+str(list(series.unique()))
        
        return isin_str
    
            
def main():
    
    df = pd.read_parquet("hplc_py/schema_cache/bounds.parquet")
    
    import ipdb; ipdb.set_trace()
    
    im = InterpetModel()
    im.gen_schema_def(df)

if __name__ == "__main__":
    main()