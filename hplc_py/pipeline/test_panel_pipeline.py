import param
import panel as pn
import polars as pl
from .preprocess_dashboard import get_bcorr_tformer, get_bcorr_viz, get_column_transformer
from 

pn.config.theme = "dark"
pipeline = pn.pipeline.Pipeline()

TIME = "time"
SIGNAL = "signal"


def ringland_dset():
    path = "tests/tests_jonathan/test_data/a0301_2021_chris_ringland_shiraz.csv"

    dset = pl.read_csv(path).select([TIME, SIGNAL])
    return dset


class Ingestion(param.Parameterized):

    signal = param.DataFrame(default=ringland_dset())
    time_col = param.String(default=TIME)
    amp_col = param.String(default=SIGNAL)

    def panel(self):
        return pn.Column(self.signal.to_pandas())

    @param.output
    def output(self):
        return self.signal, self.time_col, self.amp_col


pipeline = pn.pipeline.Pipeline()

pipeline.add_stage(
    "ingestion", Ingestion
)

pipeline.servable()


class Preprocessing(param.Parameterized):
    
    
    def output(self):
        pass
    
    def view(self):
        pass
    
    def panel(self):
        pass

class PeakMapping(param.Parameterized):

    c = param.Integer(default=6, bounds=(0, None))
    exp = param.Number(default=0.1, bounds=(0, 3))

    @param.depends("c", "exp")
    def view(self):
        out = pn.pane.LaTeX(
            "${%s}^{%s}={%.3f}$" % (self.c, self.exp, self.c**self.exp),
            styles={"font-size": "2em"},
        )
        return pn.Column(out, margin=(40, 10), styles={"background": "#f0f0f0"})

    def panel(self):
        return pn.Row(self.param, self.view)


# pipeline.add_stage("Stage 1", Preprocessing)
# pipeline.add_stage("Stage 2", PeakMapping)
