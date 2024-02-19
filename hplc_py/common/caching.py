import pandas as pd
import hashlib
import os
from cachier import cachier

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache")

def custom_param_hasher(args, kwargs):
    """
    2024-02-19 20:45:51

    In Python, mutable containers such as lists, dictionaries, DataFrames etc., are not hashable by default and must be converted to a hashable form to make function memoizing possible. This function filters kwargs based on whether they are lists, dataframes or other, and converts the aforementioned types to hashable types prior to calculating a deterministic SHA-256 hash.

    To be used with cachier decorator to memoize expensive functions such as popt factory.

    Note: will need to define a desired cache location, probably submodule by submodule.

    Notes:
    -have only tested on the following mutable objects: lists of strings, pandas dataframes. Have not tested on dicts and should raise an error.
    - excludes callables from hash because the memory address changes every run, and the memory address is included in the repr. This will be a possible source of bugs
    """
    # if lists, extract for conversion to tuples

    tupled_kwargs = {
        k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()
    }
    dframe_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, pd.DataFrame)}

    df_hash = sum(pd.util.hash_pandas_object(v).sum() for v in dframe_kwargs.values())

    callables = {k:v for k, v in tupled_kwargs.items() if callable(v)}

    hashable_args_hash = {
        k: hashlib.sha256(str(v).encode()).hexdigest()
        for k, v in tupled_kwargs.items()
        if (k not in dframe_kwargs.keys()) & (k not in callables.keys())
    }

    hash_inp = (df_hash, hashable_args_hash)
    out_hash = hashlib.sha256(str(hash_inp).encode()).hexdigest()
    
    return out_hash
