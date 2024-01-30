from .typed_dicts import InterpModelKwargs
from .interpret_model import interpret_model

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