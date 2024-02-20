from typing import Any, Iterable
import pandas as pd
import hashlib
import os
from cachier import cachier

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache")


def custom_param_hasher(args: list[Any], kwargs: dict[str, Any]) -> str:
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

    list_hash: dict[str, str] = hash_lists(kwargs)

    df_hashes: dict[str, str] = hash_dataframes(kwargs)

    callable_hashes: dict[str, str] = hash_callables(kwargs=kwargs)

    exclude_keys: list[str] = list(list_hash.keys()) + list(df_hashes.keys())

    other_args_hashes: dict[str, str] = hash_other_args(
        exclude_keys=exclude_keys, kwargs=kwargs
    )

    hashes: dict[str, str] = df_hashes | callable_hashes | other_args_hashes

    out_hash: str = sha_256_hash(hashes)

    return out_hash


def hash_other_args(
    exclude_keys: Iterable[str], kwargs: dict[str, Any]
) -> dict[str, str]:

    other_args_hashes: dict[str, str] = {
        k: sha_256_hash(v) for k, v in kwargs.items() if k not in exclude_keys
    }

    return other_args_hashes


def hash_dataframes(kwargs) -> dict[str, str]:
    df_hashes = {
        k: sha_256_hash(pd.util.hash_pandas_object(v).sum())
        for k, v in kwargs.items()
        if isinstance(v, pd.DataFrame)
    }

    return df_hashes


def hash_callables(kwargs: dict[str, Any]) -> dict[str, str]:
    """
    WARNING: this is a hack until i figure out a better method. repr of bound methods return the memory addresss of the class, which changes on every run, thus the hash changes every run, rendering the cache unusable. As the current implementation (2024-02-20 10:49:12) of `popt_factoy` passes the optimizer and fit_func as callable objects, they need to be hashable. The current work-around will use the docstring of the function as the hash input. THIS REQUIRES THE INPUT TO HAVE DOCSTRINGS.
    """
    callables = {k: v for k, v in kwargs.items() if callable(v)}

    try:
        callable_hashes = {k: sha_256_hash(v.__doc__) for k, v in callables.items()}
    except AttributeError as e:
        e.add_note(
            "this error is raised when an input func (optimizer or fit func) doesnt have a docstring. Here we are attempting to hash the docstring. Please add a docstring"
        )
        raise e
    return callable_hashes


def hash_lists(kwargs: dict[str, Any]) -> dict[str, str]:
    list_hash = {
        k: (sha_256_hash(tuple(v)) if isinstance(v, list) else v)
        for k, v in kwargs.items()
    }

    return list_hash


def sha_256_hash(x: Any) -> str:
    """
    return the SHA-256 hash string of the input. Note: object must have a __str__ or __repr__
    """
    return hashlib.sha256(str(x).encode()).hexdigest()
