import pandas as pd

class DFrameChecks:
    def _check_df(
        self,
        df: pd.DataFrame,
    ) -> None:
        if isinstance(df, pd.DataFrame):
            if df.empty:
                raise ValueError("df is empty")
        else:
            raise TypeError(f"df expected to be Dataframe, got {type(df)}\n{df}")
    
