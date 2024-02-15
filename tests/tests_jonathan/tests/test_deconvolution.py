from typing import Any, Callable, Literal, Tuple, TypeAlias

import pandera as pa
import pytest
from numpy import float64, ndarray
from pandera.typing.pandas import DataFrame
from pytest_benchmark.fixture import BenchmarkFixture

from hplc_py.deconvolve_peaks.deconvolution import (
    PeakDeconvolver,
    RSignal,
)
from hplc_py.hplc_py_typing.hplc_py_typing import (
    Params,
    Popt,
    PReport,
    PSignals,
    X_Windowed,
)
from hplc_py.map_peaks.map_peaks import PeakMapWide

Chromatogram: TypeAlias = None


@pytest.fixture
def optimizer_jax():
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
    dc: PeakDeconvolver,
    ws_: DataFrame[X_Windowed],
    params: DataFrame[Params],
    optimizer_scipy: Callable[..., Any],
    fit_func_scipy: Callable,
) -> DataFrame[Popt]:
    popt = dc._popt_factory(
        ws_,
        params,
        optimizer_scipy,
        fit_func_scipy,
    )

    return popt


@pytest.fixture
def fit_func_jax():
    import hplc_py.skewnorms.skewnorms as sk

    return sk.fit_skewnorms_jax


@pytest.fixture
def popt(
    dc: PeakDeconvolver,
    ws_: DataFrame[X_Windowed],
    params: DataFrame[Params],
    optimizer_jax: Callable[..., Tuple[ndarray[Any, Any], ndarray[Any, Any]]],
    fit_func_jax: Callable[..., Any | Literal[0]],
) -> DataFrame[Popt]:
    popt = dc._popt_factory(
        ws_,
        params,
        optimizer_jax,
        fit_func_jax,
    )

    return popt


def test_popt_factory_benchmark(
    dc: PeakDeconvolver,
    ws_: DataFrame[X_Windowed],
    params: DataFrame[Params],
    optimizer_jax,
    fit_func_jax,
    benchmark: BenchmarkFixture,
):
    benchmark(
        dc._popt_factory,
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
def test_deconvolve_peaks(
    dc: PeakDeconvolver,
    ws_: DataFrame[X_Windowed],
    peak_map: DataFrame[PeakMapWide],
    timestep: float64,
) -> None:
    dc.deconvolve_peaks(
        peak_map,
        ws_,
        timestep,
    )
