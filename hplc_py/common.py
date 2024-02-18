import distinctipy
import polars as pl


def set_peak_color_table(
    idx: pl.DataFrame,
) -> pl.DataFrame:
    """
    Set a color table indexed by the passed idx - can be 1+ columns, thus assigning a color for each group.

    Will find the unique rows in the passed idx frame and use them to assign unique colors. These are intended to be join keys for downstream color assignment.

    Returns the indexed color table.
    """
    # find unique rows in the idx

    idx_u = idx.unique()
    colors = distinctipy.get_colors(len(idx_u))

    colors = idx_u.with_columns(pl.Series("color", colors))

    return colors

