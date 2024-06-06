from abc import ABC, abstractmethod


class Report(ABC):
    """
    A base class template for pipeline reports for monitoring intermediates and output (?).
    """

    @abstractmethod
    def __init__(self):
        self._tables = None
        self._viz = None

    @property
    @abstractmethod
    def tables(self):
        return self._tables

    @tables.setter
    @abstractmethod
    def tables(self, tables):
        self._tables = tables
    
    @tables.getter
    @abstractmethod
    def tables(self):
        return self._tables
        
    @property
    @abstractmethod
    def viz(self):
        return self._viz
    
    @viz.setter
    @abstractmethod
    def viz(self, viz):
        self._viz = viz
    
    @viz.getter
    @abstractmethod
    def viz(self, viz):
        return self._viz
