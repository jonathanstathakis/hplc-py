"""
TODO:

- [x] serialise inputs:
    - [x] original signal
    - [x] unmixed_chromatograms
- [x] define target
    - [x] scores schema.

    
"""
import os
import pytest

import pandas as pd
import pandera as pa
import pandera.typing as pt

import numpy as np
import numpy.typing as npt

from hplc_py.hplc_py_typing.hplc_py_typing import *

from hplc_py.quant import Chromatogram

import pickle

class ScoresMain(pa.DataFrameModel):
    """
    An interpeted base model. Automatically generated from an input dataframe, ergo if manual modifications are made they may be lost on regeneration.
    """

    window_id: np.int64 = pa.Field(eq=[1, 2, 3, 1, 2])
    time_start: np.float64 = pa.Field(eq=[0.0, 99.32, 117.55, 6.87, 102.78])
    time_end: np.float64 = pa.Field(eq=[6.86, 102.77, 149.99, 99.31, 117.54])
    signal_area: np.float64 = pa.Field(eq=[6.843593676848658, 3.6286633406994375, 25.653215435173358, 22012.391863070363, 9998.94542314851])
    inferred_area: np.float64 = pa.Field(eq=[1.0000064024976956, 1.0000001031521522, 1.0000000002225002, 21984.982752974865, 9996.961060402295])
    signal_variance: np.float64 = pa.Field(eq=[3.829328637357519e-05, 3.3644965063619743e-31, 5.571819507195874e-30, 48.28566710621585, 145.06834273038515])
    signal_mean: np.float64 = pa.Field(eq=[0.00850595877270547, 0.007597292892194906, 0.00759729289219518, 2.380896902441359, 6.769089656837177])
    signal_fano_factor: np.float64 = pa.Field(eq=[0.004501936512607307, 4.4285465284858196e-29, 7.3339537994117525e-28, 20.280452738925394, 21.430997384390917])
    reconstruction_score: np.float64 = pa.Field(eq=[0.14612299468927253, 0.27558359904487545, 0.03898146814186073, 0.9987548327203151, 0.9998015427965411])
    window_type: np.object_ = pa.Field(eq=['interpeak', 'interpeak', 'interpeak', 'peak', 'peak'])
    applied_tolerance: np.float64 = pa.Field(eq=[0.01, 0.01, 0.01, 0.01, 0.01])
    status: np.object_ = pa.Field(eq=['needs review', 'needs review', 'needs review', 'valid', 'valid'])

    class Config:

        name="ScoresMain"
        strict=True

@pytest.fixture
def scores_main(acr):
    return acr.tables['scores']

def test_scores_main(scores_main):
    print("")
    print(scores_main)
    schema_str = interpret_model(
        scores_main, "ScoresMain", "", check_dict = {col: 'eq' for col in scores_main}
    )
    print("")
    print(schema_str)
    
def test_signal_df(
   signal_df
):
    print("")
    print(signal_df)

def test_unmixed_df(unmixed_df):
    
    print("")
    print(unmixed_df)

class TestScores:
    """
    Got the unmixed areas in the report df.
    
    - [ ] pickle the fitted chm object to speed up development iteration
    """
    
    @pytest.fixture
    def path_fitted_chm(
        self
    ):
        return os.path.join(os.getcwd(), "tests/jonathan_tests/fitted_chm.pk")
    
    @pytest.fixture
    def fitted_chm(
        self,
        chm: Chromatogram,
        amp_colname: str,
        time_colname: str,
        signal_df: pt.DataFrame[SignalDFInBase],
        
    ):
        chm.load_data(signal_df)
    
        chm.fit_peaks(
            amp_colname,
            time_colname, 
        )
        
        return chm
    
    def test_pickle_fitted_chm(
        self,
        fitted_chm: Chromatogram,
        path_fitted_chm: str,
    ):
        
        with open(path_fitted_chm, 'wb') as f:
            pickle.dump(fitted_chm,
                        f,
                        )
    
    @pytest.fixture
    def fitted_chm_pk(
        self,
        path_fitted_chm: str,
    ):
        with open(path_fitted_chm, 'rb') as f:
            fitted_chm = pickle.load(f)
        return fitted_chm
    
    @pytest.mark.xfail
    def test_chm_pickle(
        self,
        fitted_chm: Chromatogram,
        fitted_chm_pk: Chromatogram,
    ):
      assert fitted_chm == fitted_chm_pk  
        

    def test_score_df_factory(
        self,
        chm: Chromatogram,
        amp_colname: str,
        time_colname: str,
        signal_df: pt.DataFrame[SignalDFInBase],
        peak_report: pt.DataFrame[OutPeakReportBase],
    ):  
        chm.load_data(signal_df)
    
        chm.fit_peaks(
            amp_colname,
            time_colname, 
        )
        
        
        # print(chm.peak_report)
        
        # print(chm.unmixed_df.head())
        
        print(
            chm
            .unmixed_df
            .set_index('peak_idx')
            .join(
                peak_report.loc[:,['peak_idx','window_idx']]
                .set_index(
                    'peak_idx'
                ),
                how='left'
            )
            .reset_index()
            
              
              )
        
