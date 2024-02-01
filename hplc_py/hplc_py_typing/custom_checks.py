import pandera.extensions as extensions
import numpy as np


@extensions.register_check_method(statistics=["col", "stats", "atol", "rtol"])  # type: ignore
def check_stats(df, *, col: str, stats: dict, atol: float = 1e-5, rtol: float = 1e-8):
    """
    Test basic statistics of a dataframe. Ideal for validating data in large frames.

    Labels the input values as "a", expected schema values as "b". i.e. "a" is the actual, "b" is the expected as per the schema.

    Provide stats as a dict of {'statistic_name' : expected_val}.
    Currently tested for: count, mean, std, min, max

    closeness is defined using `np.isclose` which calculates closeness as:

        absolute(a-b)<=(atol + rtol * absolute(b))

    i.e. whether the distance between a and b is less than or equal to a relative ratio
    of the norm of b plus an absolute distance away from b. See docs for more info.
    https://numpy.org/doc/stable/reference/generated/numpy.allclose.html

    From stack overflow:

    > "rel_tol is a relative tolerance, it is multiplied by the greater of the magnitudes of the two arguments; as the values get larger, so does the allowed difference between them while still considering them equal."
    <https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python>

    Presumably, this can be used to lessen the weight of smaller decimal places

    :param col: the column in question
    :type col: str
    :param stats: the statistics and their expected values, as defined in the schema.
    :type stats: dict[str, float]
    :param atol: the absolute difference tolerance
    :type atol: float
    :param rtol: the relative difference tolerance
    :type rtol: float
    :raises ValueError: if any statistic doesnt match the expected value

    TODO:
    - [ ] add a detailed error message identifying which statistic failed
    """

    # statistics that I have checked to behave as expected
    valid_stats = ["count", "mean", "std", "min", "max"]

    # validate keys
    for key in stats.keys():
        if key not in valid_stats:
            raise ValueError(f"{key} is not a valid statistic, please re-enter")

    # want to lazy validate so user gets the full picture, hence calculate all then check later
    check_obj = df.rename({col: "a"}, axis=1).copy(deep=True)
    check_obj = check_obj["a"].agg(list(stats.keys())).reset_index()
    check_obj["b"] = stats.values()
    check_obj["atol"] = atol
    check_obj["atol"] = rtol
    check_obj["isclose"] = np.isclose(
        check_obj["a"],
        check_obj["b"],
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    )

    # takes all the failed stat rows and formats them to present 'a' and 'b' to the user]
    if not check_obj["isclose"].all():
        faildict = check_obj.set_index("index")[["a", "b"]].to_dict(orient="index")
        printdict = {
            k: {kk: f"{vv:.2e}" for kk, vv in v.items()} for k, v in faildict.items()
        }
        errstr = str(printdict)

        # truncate the errstr as as of 2024-01-31 14:13:29 pandera presents the errorstr in a table with ~ 150 or so character space before it messes up formatting. Hopefull they wil change this in the future
        # pandera catches any errors in custom checks and formats the message in a schema error summary alongside other fails. I think this is if `validation=lazy`.
        if len(errstr) > 150:
            errstr = errstr[:145] + "..trunc"

        raise ValueError(f"{errstr}")
    else:
        return True


@extensions.register_check_method(statistics=["col_a", "col_b"])
def col_a_less_than_col_b(df, *, col_a: str, col_b: str):
    return df[col_a] < df[col_b]


"""
PO checks
"""


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
