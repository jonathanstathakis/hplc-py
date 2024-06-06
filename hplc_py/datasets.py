import polars as pl


import os


class RinglandData:
    def __init__(self, root):
        self.root = root
        self.relpath = (
            "tests/tests_jonathan/test_data/a0301_2021_chris_ringland_shiraz.csv"
        )
        self.fullpath = os.path.join(root, self.relpath)

    def fetch(self):
        self.data = pl.read_csv(self.fullpath).rename({"": "idx"})
        return self.data


class DataSets:
    def __init__(self):
        self.root = "/Users/jonathan/hplc-py/"
        self.ringland = RinglandData(self.root)
