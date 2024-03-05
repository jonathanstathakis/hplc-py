from typing import Any, Callable, Literal, Tuple, TypeAlias

import pandera as pa
import pytest
from numpy import ndarray
from pandera.typing.pandas import DataFrame
from pytest_benchmark.fixture import BenchmarkFixture
from hplc_py.common.common_schemas import X_Schema

from hplc_py.deconvolution.deconvolution import (
    PeakDeconvolver,
    RSignal,
)
from hplc_py.map_windows.schemas import X_Windowed
from hplc_py.deconvolution.schemas import Params, Popt, PReport, PSignals
from hplc_py.deconvolution.deconvolution import popt_factory
from hplc_py.common.definitions import (
    LB_KEY,
    P0_KEY,
    P_IDX_KEY,
    PARAM_KEY,
    POPT_IDX_KEY,
    UB_KEY,
    VALUE_KEY,
    W_IDX_KEY,
    X_IDX,
    X,
)

Chromatogram: TypeAlias = None


@pytest.fixture
def optimizer_jax() -> Callable:
    from jaxfit import CurveFit

    cf = CurveFit()
    return cf.curve_fit


@pytest.fixture
def optimizer_scipy():
    from scipy.optimize import curve_fit

    return curve_fit


@pytest.fixture
def fit_func_scipy():
    import hplc_py.skewnorms.skewnorms as sk

    return sk._fit_skewnorms_scipy


@pytest.fixture
def popt_scipy(
    ws_: DataFrame[X_Windowed],
    params: DataFrame[Params],
    optimizer_scipy: Callable[..., Any],
    fit_func_scipy: Callable,
) -> DataFrame[Popt]:
    popt = popt_factory(
        X=ws_,
        params=params,
        optimizer=optimizer_scipy,
        fit_func=fit_func_scipy,
        lb_key=LB_KEY,
        optimizer_kwargs={},
        p0_key=P0_KEY,
        p_idx_key=P_IDX_KEY,
        param_key=PARAM_KEY,
        popt_idx_key=POPT_IDX_KEY,
        ub_key=UB_KEY,
        value_key=VALUE_KEY,
        verbose=True,
        w_idx_key=W_IDX_KEY,
        X_idx_key=X_IDX,
        X_key=X,
    )

    return popt


@pytest.fixture
def fit_func_jax():
    import hplc_py.skewnorms.skewnorms as sk

    return sk.fit_skewnorms_jax


@pytest.fixture
def popt(
    ws_: DataFrame[X_Windowed],
    params: DataFrame[Params],
    optimizer_jax: Callable[..., Tuple[ndarray[Any, Any], ndarray[Any, Any]]],
    fit_func_jax: Callable[..., Any | Literal[0]],
) -> DataFrame[Popt]:
    popt = popt_factory(
        X=ws_,
        params=params,
        optimizer=optimizer_jax,
        fit_func=fit_func_jax,
        lb_key=LB_KEY,
        optimizer_kwargs={},
        p0_key=P0_KEY,
        p_idx_key=P_IDX_KEY,
        param_key=PARAM_KEY,
        popt_idx_key=POPT_IDX_KEY,
        ub_key=UB_KEY,
        value_key=VALUE_KEY,
        verbose=True,
        w_idx_key=W_IDX_KEY,
        X_idx_key=X_IDX,
        X_key=X,
    )

    return popt


def test_popt_factory_benchmark(
    peak_deconvolver: PeakDeconvolver,
    ws_: DataFrame[X_Windowed],
    params: DataFrame[Params],
    optimizer_jax,
    fit_func_jax,
    benchmark: BenchmarkFixture,
):
    benchmark(
        popt_factory,
        ws_,
        params,
        optimizer_jax,
        fit_func_jax,
    )


def test_popt_factory(
    popt: DataFrame[Popt],
):
    Popt.validate(popt, lazy=True)

    return None


def test_store_popt(
    popt: DataFrame[Popt],
    popt_parqpath: Literal["/Users/jonathan/hplc-py/tests/jonathan_tests/assch…"],
):
    popt.to_parquet(popt_parqpath)


def test_construct_peak_signals(psignals: DataFrame[PSignals]) -> None:
    PSignals.validate(psignals, lazy=True)

    return None


@pa.check_types
def test_reconstruct_signal(
    r_signal: DataFrame[RSignal],
):
    pass


def test_peak_report(
    peak_report: DataFrame[PReport],
):
    PReport.validate(peak_report, lazy=True)

    return None


def dc() -> PeakDeconvolver:
    dc = PeakDeconvolver()
    return dc


@pa.check_types
def test_deconvolve_peaks(pdc_tform: PeakDeconvolver) -> None:
    breakpoint()
    pass


def plot_overlay(df):
    import polars as pl

    plot_obj = df.pipe(pl.from_pandas).plot(x="X_idx", y=["X", "recon"])

    import hvplot

    hvplot.show(plot_obj)
