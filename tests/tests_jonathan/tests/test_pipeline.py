from hplc_py.pipeline.pipeline import DeconvolutionPipeline
import pytest
import pandera as pa
from pandera.typing import DataFrame
from hplc_py.common.common_schemas import RawData
import pandas as pd  

def test_pipeline(
  asschrom_dset: pd.DataFrame,
):  
    pipeline = DeconvolutionPipeline()
    
    pipeline.run(
      data=asschrom_dset,
      key_time='x',
      key_amp='y',
      )
    