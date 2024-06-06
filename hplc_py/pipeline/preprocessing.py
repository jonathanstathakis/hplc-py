from hplc_py.baseline_correction import baseline_correction
from typing import Any
import pandas as pd
import polars as pl
from pandera.typing import DataFrame
from hplc_py.baseline_correction import definitions as bcorr_defs
from hplc_py.common import common_schemas as com_schs
from hplc_py.transformer_abc import transformer_abc

# @cachier(hash_func=caching.custom_param_hasher, cache_dir=caching.CACHE_PATH)


class PreprocessData:
    def __init__(
        self,
        correct_baseline: bool = True,
        bcorr_kwargs: bcorr_defs.BlineKwargs = bcorr_defs.BlineKwargs(),
    ):
        self.correct_baseline_flag = correct_baseline
        self.bcorr_kwargs = bcorr_kwargs
        self._reports = {}

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        self.X = X

    def transform(self):

        if self.correct_baseline_flag:
            try:
                bcorr_out = self.X.pipe(
                    self.correct_baseline,
                    **self.bcorr_kwargs,
                )
            except Exception as e:
                raise e

            self.X_corrected: pl.DataFrame = bcorr_out.filter(
                pl.col("signal") == "bcorr__X"
            ).select(
                pl.col("idx"),
                pl.col("amp").alias("X_corrected"),
            )

            # TODO: add other preprocessing methods

            X_processed = self.X_corrected.to_pandas()

            return X_processed
        else:
            return self.X

    def set_report(self):
        pass

    @property
    def report(self):
        """
        Returns a dict of keys:
        - "baseline_report":
            "table":
            a table of the input signal, background signal, and corrected signal
            "viz":
            a visualisation of the above signals
        """
        return self._report

    def correct_baseline(
        self,
        data: DataFrame[com_schs.X_Schema],
        n_iter: int = 5,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, Any]:
        """
        TODO: ringland optimal n_iter (identified manually) is 72 for first baseline
        correct, 50 for second. add this to a hyperperameter store
        """

        X = data.pipe(pl.from_pandas).select("X")

        bcorr = baseline_correction.SNIPBCorr(
            n_iter=n_iter,
            verbose=verbose,
        )

        bcorr.fit_transform(
            X=X,
        )

        signals = bcorr.signals_

        return signals
