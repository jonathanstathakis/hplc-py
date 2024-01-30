import pandera.extensions as extensions


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
        if key not in valid_stats:
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