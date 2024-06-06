from abc import ABC, abstractmethod
from hplc_py.reports import report
from typing import Self


class Transformer(ABC):
    """
    A template transformer class defining a common interface. Each transformer will require:
        - an `__init__` defined which will take data invariant settings as kwrgs  and transform specific kwargs
        - fit which will take the data as 'X', and kwargs relevant to data ingestion (note 'y' is included to match scikit-learn API)
        - the transform method, which takes no args
        - `fit_transform` which wil lcall both.
        - report, which will return something to be used as a report, probably a dict, containing at least a table of data and a viz.
    """

    @abstractmethod
    def __init__(self, **kwargs):

        self._report = report.Report()
        pass

    @abstractmethod
    def fit(self, X, y=None, **kwargs)->Self:
        self.X = X
        pass

    @abstractmethod
    def transform(self):
        X_transform = None
        
        return X_transform
    
    @abstractmethod
    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        tform = self.transform()
        return tform

    @abstractmethod
    def set_report(self):
        """
        Function to handle setting the report attribute tables and viz
        """
        pass

    @property
    @abstractmethod
    def report(self):
        return self._report

    @report.getter
    def report(self):
        return self._report

    @report.setter
    def report(self, tables, viz):
        self._report.tables = tables
        self._report.viz = viz
