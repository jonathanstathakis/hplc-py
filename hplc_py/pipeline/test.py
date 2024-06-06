import panel as pn

col = pn.Column("test")

col.servable()


class DashBoard:
    def __init__(self):
        self.col = pn.Column("oop test")
        self.col.servable()

    def add_to_col(self, x):
        self.col.append(x)

    def remove_by_idx(self, idx: int):
        self.col.pop(idx)


dboard = DashBoard()
dboard.add_to_col("test append")
dboard.remove_by_idx(1)
