from typing import Any

import pandas as pd
import pandera as pa
import pandera.typing as pt

import numpy.typing as npt
import numpy as np


class SignalDFIn(pa.DataFrameModel):
    """
    The base signal, with time and amplitude directions
    """

    time: np.float64
    amp: np.float64


class OutPeakDF_Base(pa.DataFrameModel):
    """
    Contains information about each detected peak, used for profiling
    """

    peak_idx: pd.Int64Dtype = pa.Field()
    time_idx: pd.Int64Dtype = pa.Field(
        coerce=False
    )  # the time idx values corresponding to the peak maxima location
    peak_prom: pd.Float64Dtype = pa.Field(coerce=False)
    whh: pd.Int64Dtype = pa.Field(coerce=False)
    whhh: pd.Int64Dtype = pa.Field(coerce=False)
    whh_left: pd.Int64Dtype = pa.Field(coerce=False)
    whh_right: pd.Int64Dtype = pa.Field(coerce=False)
    rl_width: pd.Int64Dtype = pa.Field(coerce=False)
    rl_wh: pd.Int64Dtype = pa.Field(coerce=False)
    rl_left: pd.Int64Dtype = pa.Field(coerce=False)
    rl_right: pd.Int64Dtype = pa.Field(coerce=False)

    class Config:
        name='OutPeakDF'
        strict=False

    @pa.dataframe_check
    def check_null(cls, df: pd.DataFrame) -> bool:
        return df.shape[0] > 0


class OutPeakDF_ManyPeaks(OutPeakDF_Base):
    """
    commented out fields are specific to 'test_many_peaks.csv'. Until I find a method of parametrizing
    """

    time_idx: pd.Int64Dtype = pa.Field(
        isin=[1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
    )
    peak_prom: pd.Float64Dtype = pa.Field(
        isin=[
            0.9221250775321379,
            0.9129314313007107,
            0.9131520182120347,
            0.9135276076962014,
            0.9140168870790715,
            0.9138066553780264,
            0.9133958162236926,
            0.9131460523698959,
            0.9130556194514102,
            1.0,
        ],
    )

    whh: pd.Int64Dtype = pa.Field(isin=[220.0, 221.0, 235.0])
    whhh: pd.Int64Dtype = pa.Field(isin=[0])
    whh_left: pd.Int64Dtype = pa.Field(
        isin=[888, 1389, 1889, 2389, 2889, 3389, 3889, 4389, 4889, 5382]
    )
    whh_right: pd.Int64Dtype = pa.Field(
        isin=[1110, 1610, 2110, 2610, 3110, 3610, 4110, 4610, 5110, 5617]
    )

    rl_width: pd.Int64Dtype = pa.Field(
        isin=[489.0, 490.0, 2475.0, 492.0, 496.0, 494.0, 491.0, 5233.0]
    )
    rl_wh: pd.Int64Dtype = pa.Field(isin=[0])
    rl_left: pd.Int64Dtype = pa.Field(
        isin=[774, 1250, 1750, 2250, 2750, 3255, 3758, 4259, 4760, 636]
    )
    rl_right: pd.Int64Dtype = pa.Field(
        isin=[3249, 1739, 2240, 2742, 3246, 3749, 4250, 4749, 5250, 5869]
    )


class OutPeakDF_AssChrom(OutPeakDF_Base):
    """
    commented out fields are specific to 'test_many_peaks.csv'. Until I find a method of parametrizing
    """

    time_idx: pd.Int64Dtype = pa.Field(isin=[1507, 1899, 8000, 11000])
    peak_prom: pd.Float64Dtype = pa.Field(
        nullable=True,
        isin=[15.07, 18.99, 80.0, 110.0],
    )
    peak_prom: pd.Float64Dtype = pa.Field(
        nullable=True,
        isin=[1.0, 0.0760169246981745, 0.06228727805122132, 0.9343909891207713],
    )

    whh: pd.Int64Dtype = pa.Field(isin=[270.0, 172.0, 706.0, 235.0])
    whhh: pd.Int64Dtype = pa.Field(isin=[0])
    whh_left: pd.Int64Dtype = pa.Field(isin=[1385, 1809, 7646, 10882])
    whh_right: pd.Int64Dtype = pa.Field(isin=[1655, 1982, 8353, 11117])

    rl_width: pd.Int64Dtype = pa.Field(isin=[8721.0, 282.0, 2921.0, 1409.0])
    rl_wh: pd.Int64Dtype = pa.Field(isin=[0])
    rl_left: pd.Int64Dtype = pa.Field(isin=[687, 1736, 6455, 10297])
    rl_right: pd.Int64Dtype = pa.Field(isin=[9408, 2018, 9376, 11706])


class OutSignalDF_Base(pa.DataFrameModel):
    time_idx: pd.Int64Dtype = pa.Field(coerce=False)
    amp: pd.Float64Dtype = pa.Field(coerce=False)
    norm_amp: pd.Float64Dtype = pa.Field(coerce=False)


class OutSignalDF_ManyPeaks(OutSignalDF_Base):
    time_idx: pd.Int64Dtype = pa.Field(ge=0, lt=7000)
    amp: pd.Float64Dtype = pa.Field(ge=6.425446e-48, le=3.989453e01)
    norm_amp: pd.Float64Dtype = pa.Field(ge=0, le=1)


class OutSignalDF_AssChrom(OutSignalDF_Base):
    time_idx: pd.Int64Dtype = pa.Field(ge=0, lt=15000)
    amp: pd.Float64Dtype = pa.Field(ge=-0.029838, le=42.690122)
    norm_amp: pd.Float64Dtype = pa.Field(ge=0, le=1)


class OutWindowDF_Base(pa.DataFrameModel):
    """
    Contains a recording of each window in the Chromatogram labeled with an ID and type
    with a time index corresponding to the time values in the time array.

    Spans the length of the chromatogram signal
    """

    time_idx: pd.Int64Dtype = pa.Field(coerce=False)
    window_idx: pd.Int64Dtype = pa.Field(coerce=False)
    window_type: str  # either 'peak' or 'np.int64erpeak'


class OutWindowDF_ManyPeaks(OutWindowDF_Base):
    time_idx: pd.Int64Dtype = pa.Field(ge=636, le=5868)
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])


class OutWindowDF_AssChrom(OutWindowDF_Base):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1, 2])
    time_idx: pd.Int64Dtype = pa.Field(ge=687, le=11705)


class OutInitialGuessBase(pa.DataFrameModel):
    """
    The DF containing the initial guesses of each peak by window, created by `PeakDeconvolver.p0_factory`
    """

    window_idx: pd.Int64Dtype
    peak_idx: pd.Int64Dtype
    param: pd.CategoricalDtype
    p0: pd.Float64Dtype


class OutInitialGuessManyPeaks(OutInitialGuessBase):
    """
    The DF containing the initial guesses of each peak by window, created by `PeakDeconvolver.p0_factory`
    """

    pass
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    p0: pd.Float64Dtype = pa.Field(
        isin=[
            0.0,
            39.89452538404621,
            220.0,
            39.89437671209474,
            1000.0,
            221.0,
            1500.0,
            2000.0,
            2500.0,
            3000.0,
            3500.0,
            4000.0,
            4500.0,
            5000.0,
            5500.0,
            235.0,
        ]
    )


class OutInitialGuessAssChrom(OutInitialGuessBase):
    """
    The DF containing the initial guesses of each peak by window, created by `PeakDeconvolver.p0_factory`
    """

    window_idx: pd.Int64Dtype = pa.Field(isin=[1, 2])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    p0: pd.Float64Dtype = pa.Field(
        isin=[
            0.0,
            42.69012166052291,
            1507.0,
            270.0,
            19.96079318035132,
            1899.0,
            172.0,
            2.659615202676218,
            8000.0,
            706.0,
            39.89422804014327,
            11000.0,
            235.0,
        ]
    )


class OutDefaultBoundsBase(pa.DataFrameModel):
    window_idx: pd.Int64Dtype
    peak_idx: pd.Int64Dtype
    param: pd.CategoricalDtype
    lb: pd.Float64Dtype
    ub: pd.Float64Dtype


class OutDefaultBoundsManyPeaks(OutDefaultBoundsBase):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    lb: pd.Float64Dtype = pa.Field(
        isin=[
            636.0,
            -np.inf,
            0.009999999999999998,
            3.9894525384046213,
            3.9894376712094743,
        ]
    )
    ub: pd.Float64Dtype = pa.Field(
        isin=[5868.0, np.inf, 2616.0, 398.9452538404621, 398.9437671209474]
    )


class OutDefaultBoundsAssChrom(OutDefaultBoundsBase):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1, 2])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    lb: pd.Float64Dtype = pa.Field(
        isin=[
            -np.inf,
            0.01,
            687.0,
            4.269012166052291,
            1.9960793180351322,
            0.2659615202676218,
            3.989422804014327,
            10297.0,
        ]
    )
    ub: pd.Float64Dtype = pa.Field(
        isin=[
            np.inf,
            9407.0,
            4360.0,
            426.90121660522914,
            199.6079318035132,
            26.59615202676218,
            398.9422804014327,
            11705.0,
            704.0,
        ]
    )


class OutWindowedSignalBase(OutSignalDF_Base):
    """
    The signal DF with the addition of a window ID column
    """

    window_idx: pd.Int64Dtype


class OutWindowedSignalManyPeaks(OutSignalDF_ManyPeaks):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])


class OutWindowedSignalAssChrom(OutSignalDF_AssChrom):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1, 2])


class OutParamsBase(pa.DataFrameModel):
    window_idx: pd.Int64Dtype
    peak_idx: pd.Int64Dtype
    param: pd.CategoricalDtype
    p0: pd.Float64Dtype
    lb: pd.Float64Dtype
    ub: pd.Float64Dtype
    inbounds: bool


class OutParamManyPeaks(OutParamsBase):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    p0: pd.Float64Dtype = pa.Field(
        isin=[0.0, 39.89452538404621, 110.0, 39.89437671209474, 1000.0, 110.5, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 117.5]
    )
    lb: pd.Float64Dtype = pa.Field(
        isin=[636.0, 1.0, -np.inf, 3.9894525384046213, 3.9894376712094743]
    )
    ub: pd.Float64Dtype = pa.Field(
        isin=[5868.0, 2616.0, np.inf, 398.9452538404621, 398.9437671209474]
    )
    inbounds: bool = pa.Field(isin=[True])


class OutParamAssChrom(OutParamsBase):
    window_idx: pd.Int64Dtype = pa.Field(isin=[1, 2])
    peak_idx: pd.Int64Dtype = pa.Field(isin=[0, 1, 2, 3])
    param: pd.CategoricalDtype = pa.Field(isin=["amp", "loc", "whh", "skew"])
    p0: pd.Float64Dtype = pa.Field(
        isin=[0.0, 42.69012166052291, 1507.0, 135.0, 19.96079318035132, 1899.0, 86.0, 2.659615202676218, 8000.0, 353.0, 39.89422804014327, 11000.0, 117.5]
    )
    lb: pd.Float64Dtype = pa.Field(
        isin=[1.0, -np.inf, 687.0, 4.269012166052291, 1.9960793180351322, 0.2659615202676218, 3.989422804014327, 10297.0]
    )
    ub: pd.Float64Dtype = pa.Field(
        isin=[
            np.inf,
            9407.0,
            4360.0,
            426.90121660522914,
            199.6079318035132,
            26.59615202676218,
            398.9422804014327,
            11705.0,
            704.0,
        ]
    )
    inbounds: bool = pa.Field(isin=[True])


class OutPoptBase(pa.DataFrameModel):
    pass


class OutPoptManyPeaks(OutPoptBase):
    pass


class OutPoptAssChrom(OutPoptBase):
    pass


def isArrayLike(x: Any):
    if not any(x):
        raise ValueError("x is None")

    if not hasattr(x, "__array__"):
        return False
    else:
        return True


class OutReconDFBase(pa.DataFrameModel):
    '''
    it is always better to have a set number of columns and very the length than it is
    to have a varying number of columns. Thus it is best to store ReconDF in long form
    then pivot where necessary. Of course the most space efficient form will be one that
    does not repeat the times. That in itself is a side-study that we do not have time for.
    '''
    
    peak_idx: pd.Int64Dtype
    time_idx: pd.Int64Dtype
    unmixed_amp: pd.Float64Dtype
    
class OutReconDFManyPeaks(OutReconDFBase):
    pass

class OutReconDFAssChrom(OutReconDFBase):
    pass

class OutPeakReportBase(OutPoptBase):
    unmixed_area: pd.Float64Dtype
    unmixed_maxima: pd.Float64Dtype
    tbl_name: pd.StringDtype = pa.Field(eq='peak_report')
    
class OutPeakReportManyPeaks(OutPeakReportBase):
    pass

class OutPeakReportAssChrom(OutPeakReportBase):
    pass