"""
TODO:

- [x] serialise inputs:
    - [x] original signal
    - [x] unmixed_chromatograms
- [x] define target
    - [x] scores schema.

    
"""
import os
from typing import Any

from pandas.testing import assert_frame_equal


from pandera.typing.pandas import DataFrame
import pytest

import pandas as pd

import pandas.testing as pdt

import pandera as pa
import pandera.typing as pt

import numpy as np
import numpy.typing as npt

from hplc_py.hplc_py_typing.hplc_py_typing import *
from hplc_py.hplc_py_typing.hplc_py_typing import Recon

from hplc_py.quant import Chromatogram
from .test_jonathan import AssChromResults
import pickle

import hplc
pd.options.display.max_columns = 50


@pytest.fixture
def main_fitted_chm():

    import hplc_py as hplc
    pkpth = "/Users/jonathan/hplc-py/tests/jonathan_tests/fitted_chm_main.pk"

    with open(pkpth, 'rb') as f:
        main_fitted_chm = pickle.load(f)

    return main_fitted_chm


def adapt_ms_df(
    df: DataFrame,
    ):
    if not ("window_idx" in df.columns) & ("window_id" in df.columns):
        df = df.rename({"window_id":"window_idx"},axis=1)
        
    ms_to_mys_mapping = {
        "signal_area": "area_amp_mixed",
        "inferred_area": "area_amp_unmixed",
        "signal_variance": "var_amp_unmixed",
        "reconstruction_score": "score",
        "signal_fano_factor": "mixed_fano",
    }

    df_ = df.rename(ms_to_mys_mapping, axis=1, errors="raise")

    df_["sw_idx"] = df_.groupby(["time_start"]).ngroup() + 1
    df_ = df_.loc[
        :,
        [
            "sw_idx",
            "window_type",
            "window_idx",
            "time_start",
            "time_end",
            "area_amp_mixed",
            "area_amp_unmixed",
            "var_amp_unmixed",
            "score",
            "mixed_fano",
            "applied_tolerance",
            "status",
        ],
    ]
    df_ = df_.sort_values("time_start").reset_index(drop=True)

    df_ = df_.reset_index(drop=True)
    return df_
    

@pytest.fixture
def m_sc_df(main_fitted_chm: hplc.quant.Chromatogram):
    ms_df = main_fitted_chm.assess_fit()
    adapted_ms_df = adapt_ms_df(ms_df)
    return adapted_ms_df


def test_ms_df_exec(m_sc_df: pd.DataFrame):
    pass

@pytest.fixture
def m_amp_recon(
    main_fitted_chm: hplc.quant.Chromatogram,
):  
    """
    The peak_props dict does not contain information regarding x...
    """
    
    recon = pd.DataFrame(
        {p: v for p, v in enumerate(main_fitted_chm.unmixed_chromatograms)}
        )
    
    recon = recon.rename_axis(columns='time_idx', index='peak_idx')
    recon = recon.T
    
    recon = pd.Series(
        np.sum(recon.to_numpy(),axis=1),
        name='amp_unmixed',
        )

    return recon


def test_m_amp_recon_exec(
    m_amp_recon
):
    pass

@pytest.fixture
def m_ws_df(
    main_fitted_chm: Chromatogram,
    m_amp_recon: DataFrame,
)->DataFrame:
    m_ws_df = main_fitted_chm.window_df.copy(deep=True)
    
    m_ws_df = m_ws_df.rename({"x":"time",
                              "y_corrected":"amp_mixed",
                              "window_id":"window_idx",
                              },axis=1)
    
    m_ws_df = m_ws_df.set_index(['window_type','window_idx','time_idx','time']).reset_index()
    m_ws_df = m_ws_df.drop(['y','estimated_background','time_idx'],axis=1, errors='raise')
    
    m_ws_df = pd.concat([m_ws_df, m_amp_recon],axis=1)
    
    m_ws_df = m_ws_df.astype({
        "window_type":pd.StringDtype(),
        'window_idx':pd.Int64Dtype(),
        "time":pd.Float64Dtype(),
        "amp_mixed":pd.Float64Dtype(),
        "amp_unmixed":pd.Float64Dtype(),
    })
    return m_ws_df

def test_main_w_df_exec(
    m_ws_df: DataFrame,
):
    pass
    

def test_signal_df_exec(signal_df: DataFrame):
    pass


def test_unmixed_df_exec(unmixed_df: Any):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.relplot(unmixed_df, x='time',y='unmixed_amp',hue='peak_idx', kind='line')
    plt.show()
    
    pass


class TestScores:
    """
    Got the unmixed areas in the report df.

    - [ ] pickle the fitted chm object to speed up development iteration
    """

    @pytest.fixture
    def path_fitted_chm(self):
        return os.path.join(os.getcwd(), "tests/jonathan_tests/fitted_chm.pk")

    def test_pickle_fitted_chm(
        self,
        fitted_chm: Chromatogram,
        path_fitted_chm: str,
    ):
        with open(path_fitted_chm, "wb") as f:
            pickle.dump(
                fitted_chm,
                f,
            )

    @pytest.fixture
    def fitted_chm_pk(
        self,
        path_fitted_chm: str,
    ):
        with open(path_fitted_chm, "rb") as f:
            fitted_chm = pickle.load(f)
        return fitted_chm

    @pytest.mark.xfail
    def test_chm_pickle(
        self,
        fitted_chm: Chromatogram,
        fitted_chm_pk: Chromatogram,
    ):
        assert fitted_chm == fitted_chm_pk

    @pytest.fixture
    def windowed_signal_df(
        self,
        fitted_chm_pk: Chromatogram,
    ) -> pt.DataFrame:
        windowed_signal_df = (
            fitted_chm_pk._signal_df.set_index("time_idx")
            .join(
                fitted_chm_pk.window_df.set_index("time_idx").loc[
                    :, ["window_type", "sw_idx", "window_idx"]
                ],
                on="time_idx",
                how="left",
            )
            .reset_index()
        )
        return windowed_signal_df

    @pytest.fixture
    def amp_col(self):
        return "amp_corrected"

    @pytest.fixture
    def score_df(
        self,
        unmixed_df: DataFrame[Recon],
        signal_df: DataFrame,
        window_df: DataFrame[Any],
        chm: Chromatogram,
    ) -> pt.DataFrame:
        score_df = chm._fitassess._score_df_factory(signal_df, unmixed_df, window_df)

        return score_df
    
    def test_score_df_exec(
        self,
        score_df: DataFrame,
    ):
        print("")
        print(score_df)
        
    
    @pytest.fixture
    def m_sc_df(
        self,
        main_fitted_chm,
    ):
        m_sc_df = main_fitted_chm._score_reconstruction()
        
        m_sc_df = m_sc_df.set_index(['window_type','window_id']).reset_index()
        
        m_sc_df = m_sc_df.rename({"window_id":"window_idx"},axis=1)
        
        m_sc_df = m_sc_df.astype({
            "window_type":pd.StringDtype(),
            "window_idx": pd.Int64Dtype(),
            **{col: pd.Float64Dtype() for col in m_sc_df if pd.api.types.is_float_dtype(m_sc_df[col])},
        })
        return m_sc_df
    
    def test_m_sc_df_exec(
        self,
        m_sc_df: DataFrame,
    ):
        print("")
        print(m_sc_df.dtypes)
        
    def test_compare_scores(
        self,
        score_df: DataFrame,
        m_sc_df: DataFrame,
    ):
        print("")
        
        # pdt.assert_frame_equal(score_df, m_sc_df, check_exact=True)
        
        for col in score_df:
            if pd.api.types.is_numeric_dtype(score_df[col]):
                
                print(col, ":",(score_df[col].to_numpy()-m_sc_df[col].to_numpy()).sum())
            else:
                pass
        
    def test_score_df_factory_exec(
        self,
        score_df: DataFrame,
    ) -> None:
        
        print(f"\n{score_df}")
        pass
    
    @pytest.fixture
    def ws_df(
        self,
        chm: Chromatogram,
        window_df: DataFrame,
        signal_df: DataFrame,
        unmixed_df: DataFrame,
    )-> DataFrame:
        
        ws_df = chm._fitassess.prep_ws_df(signal_df, unmixed_df, window_df)
        
        return ws_df
    
    def test_ws_df_exec(
        self,
        ws_df: DataFrame,
    ):
        pass
        
    
    def test_ws_df_compare(
        self,
        ws_df: DataFrame,
        m_ws_df: DataFrame,
    ):
        left_df = ws_df
        right_df = m_ws_df
        try:   
            assert_frame_equal(left_df, right_df)
        except Exception as e:
            
            err_str = str(e)
            err_str += "\n"
            
            cols = f"['left']: {left_df.columns}\n['right']: {right_df.columns}"
            
            err_str +=  cols
            err_str += "\n"
            
            dtypes =f"['left']: {left_df.dtypes}\n['right']: {right_df.dtypes}"
            err_str +="\n"
            err_str += dtypes
            
            raise AssertionError(err_str)
    
    @pytest.fixture
    def rtol(
        self,
    ):
        return 1E-2
    
    def test_assign_tolpass_exec(
        self,
        chm: Chromatogram,
        score_df: DataFrame,
        groups: list[str],
        rtol: float,
    ):
        chm._fitassess.assign_tolpass(score_df, groups, rtol)
        
    
    @pytest.fixture
    def main_fit_report(
        self,
        main_fitted_chm: hplc.quant.Chromatogram,
    ):
        report = main_fitted_chm.assess_fit()
        return report
    
    def test_main_fit_report_exec(
        self,
        main_fit_report: DataFrame,
    ):
        print("")
        print(main_fit_report)
    
    @pytest.fixture
    def ftol(self):
        return 1E-2
    
    @pytest.fixture
    def groups(self):
        return ['window_type','window_idx']
    
    def test_assign_fanopass_exec(
        self,
        chm: Chromatogram,
        score_df: DataFrame,
        groups: list[str],
        ftol: float,
    ):
        score_df_ = chm._fitassess.assign_fanopass(score_df, groups, ftol)
    
    @pytest.fixture
    def score_df_with_passes(
        self,
        chm: Chromatogram,
        score_df: DataFrame,
        groups: list[str],
        ftol: float,
        rtol: float,
    ):
        score_df_ = chm._fitassess.assign_tolpass(score_df, groups, rtol)
        score_df_ = chm._fitassess.assign_fanopass(score_df, groups, ftol)
        return score_df_

    def test_score_df_with_passes(
        self,
        score_df_with_passes: DataFrame,
    ):
        print("")
        print(score_df_with_passes)
        
    @pytest.fixture
    def score_df_with_status(
        self,
        chm: Chromatogram,
        score_df_with_passes: DataFrame,
    ):
        return chm._fitassess.assign_status(score_df_with_passes)

    def test_score_df_with_status(
        self,
        score_df_with_status: DataFrame,
    ):
        print("")
        print(score_df_with_status)
        
        
        
    
    @pytest.fixture
    def fit_report(
        self,
        chm: Chromatogram,
        signal_df: DataFrame,
        unmixed_df: DataFrame,
        peak_report: DataFrame,
        window_df: DataFrame,
        rtol: float,
    )->DataFrame:
        
        fit_report = chm._fitassess.assess_fit(
            signal_df,
            unmixed_df,
            window_df,
            rtol,
        )
        return fit_report
    
    def test_fit_report_exec(
        self, 
        fit_report: DataFrame,
    ):
        pass
    
    def test_report_card_main(
        self,
        main_fitted_chm: hplc.quant.Chromatogram,
        score_df: DataFrame,
    ):
        main_fitted_chm.assess_fit()
    
    def test_termcolors(
        self
    ):
        import termcolor
        
        termcolor.cprint("test", color='blue', on_color='white')