import pandera as pa


class PanderaSchemaMethods:
    """
    A collection of helper methods to use Pandera DataFrame Models as centralised control units for dataframe formatting - column names, ordering, and datatypes.
    """

    def get_schema_colorder(self, schema: pa.DataFrameModel):
        return list(schema.to_schema().columns.keys())
