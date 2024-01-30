"""
Contains common IO validation methods
"""
import pandas as pd
from numpy import float64
from numpy import int64
from pandera.typing import Series
from typing import Any, Optional

class IOValid:
        
    def check_scalar_is_type(
        self,
        check_obj,
        type,
        name: Optional[str] = "",
    ):
        """
        Validation of scalar input variables such as str, int and float objects. Input the check_obj and its expected type to check if the object is of that type. For containers such as lists, numpy arrays, pandas Series etc, use `check_container_is_type



        :param check_obj: the check object. Can be used for any object to check its surface level type.
        :type check_obj: any object
        :param type: the expected datatype of the `check_obj`
        :type type: any type
        :param name: an optional name to give the check_obj to include in the error message.
        :type name: str
        :raises TypeError: if `check_obj` is not an instance of `type`
        """
        if not isinstance(check_obj, type):
            raise TypeError(f"Expected input {name} to be {type}, got {type(check_obj)}")
        
    def check_container_is_type(
        self,
        check_obj,
        array_type,
        element_type,
        name: Optional[str]="",
    ):
        """
        Validation of containers and their element datatypes. Hasnt been tested thoroughly, but will work on numpy arrays, pandas Series and Polars Series. Anything that possesses a `.dtype` attribute that can be compared to the input `element_type`.

        :param check_obj: the container object to be validated
        :type check_obj: any container object, atm numpy array, pandas Series, polars Series
        :param array_type: the expected type of the container object
        :type array_type: any container object type, confirmed on numpy array, pandas Series, polars Series
        :param element_type: the type of the elements within the container. Expect 1 datatype per container, hence all elements the same datatype.
        :type element_type: any type
        :param name: an optional name to give the check_obj to include in the error message.
        :type name: str
        :raises TypeError: if `check_obj` doesnt match `array_type`
        :raises TypeError: if `check_obj` element dtype does not match `element_type`
        """
        if isinstance(check_obj, array_type):
            if not check_obj.dtype == element_type:
                    raise TypeError(
                        f"Expected input to be of type {element_type}, got {check_obj.dtype}"
                    )
            else:
                pass
        else:
            raise TypeError(
                f"Expected input {name} to be of type {array_type}, got {type(check_obj)}"
            )

    def _check_df(
        self,
        check_obj: pd.DataFrame,
    ) -> None:
        """
        Check if check_obj is a pandas DataFrame and if it is empty. Necessary to check if it is a pandas DataFrame first because otherwise `.empty` could cause an error.

        :param df: An object expected to be a non-empty pandas DataFrame
        :type df: pd.DataFrame
        :raises ValueError: if the input DataFrame is empty
        :raises TypeError: if the input is not a pandas DataFrame
        """
        if isinstance(check_obj, pd.DataFrame):
            if check_obj.empty:
                raise ValueError("df is empty")
        else:
            raise TypeError(f"df expected to be Dataframe, got {type(check_obj)}\n{check_obj}")


    def _check_keys_in_index(
            self,
            keys: list[Any],
            index: pd.Index,
        ) -> None:
        """
        Check if a given list of keys are valid for a given Pandas index.

        :param keys: a list of string keys expected to be in the `index`
        :type keys: list[Any]
        :param index: A pandas Series or DataFrame Index object, i.e. rows or columns in which we expect `keys` to be present
        :type index: pd.Index
        :raises ValueError: if `keys` are not in `index`. 
        """
        keys_: Series[Any] = Series(keys) #type: ignore

        if not (key_mask := keys_.isin(index)).any():
            raise ValueError(
                f"The following provided keys are not in index: {keys_[~key_mask].tolist()}\npossible keys: {index}"
            )
