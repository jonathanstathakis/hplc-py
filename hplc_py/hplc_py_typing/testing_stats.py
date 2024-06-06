import pandera as pa
import pandera.typing as pt
import pandera.extensions as extensions
import pandas as pd

# count
# mean
# std
# min
# max

df = pd.DataFrame({"col1": [1, 2, 3], "col2": [10, 15, 20]})

print(df["col1"].describe()[["count", "mean", "std", "min", "max"]].to_dict())


@extensions.register_check_method(statistics=["col", "stats"])  # type: ignore
def check_stats(df, *, col: str, stats: dict):
    """
    Provide stats as a dict of {'statistic_name' : expected_val}.
    Currently tested for: count, mean, std, min, max
    """
    # statistics that I have checked to behave as expected
    valid_stats = ["count", "mean", "std", "min", "max"]

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


class Schema(pa.DataFrameModel):
    col1: int
    col2: int

    class Config:
        _col1_dict = {
            "col": "col1",
            "stats": {
                "count": 3.0,
                "mean": 2.0,
                "std": 1.0,
                "min": 1.0,
                "max": 3.0,
            },
        }
        # _col2_dict = {"key":'col2', "x":15}
        check_stats = _col1_dict
        # check_stats = _col2_dict


try:
    Schema(df)
except Exception as e:
    print(e)
else:
    print("success!")
