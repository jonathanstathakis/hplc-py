import polars as pl

class Report():
    """
    A base class template for transformer reports for monitoring intermediates and output (?).
    """
    
    def __init__(self, transformer_name: str):
        self.transformer_name = transformer_name # used for report
        self._tables = pl.DataFrame()
        self._viz = None

    @property
    def tables(self):
        return self._tables

    @tables.setter
    def tables(self, tables):
        self._tables = tables
    
    @tables.getter
    def tables(self):
        return self._tables
        
    @property
    def viz(self):
        return self._viz
    
    @viz.setter
    def viz(self, viz):
        self._viz = viz
    
    @viz.getter
    def viz(self):
        return self._viz
    
    def __str__(self):
        """
        Display the head of each table and the string representation of the vizzes. Need
        to handle both singular object and lists.
        """
        # tables
        out_str = f"##### {self.transformer_name} #####\n"
        
        if isinstance(self.tables, list):
            tbl_str = f"{[table.head() for table in self.tables]}"
        else:
            tbl_str = f"{str(self.tables.head())}"
        
        # viz
        
        if isinstance(self.viz, list):
            viz_str = f"{[str(viz) for viz in self.viz]}"
        else:
            viz_str = str(self.viz)
            
        out_str += f"{tbl_str}\n\n{viz_str}"
        
        return out_str
