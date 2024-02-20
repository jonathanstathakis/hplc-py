from numpy.typing import NDArray
import scipy
import numpy as np
from numpy import float64

import jax.numpy as jnp
import jax.scipy as jscipy


def _compute_skewnorm_jax(
    x,
    params,
):
    amp, loc, scale, alpha = params

    x_minus_loc = x - loc

    _x = alpha * x_minus_loc / scale

    scale_sq = scale**2

    norm = jnp.sqrt(2 * jnp.pi * scale_sq) ** -1 * jnp.exp(
        -(x_minus_loc**2) / (2 * scale_sq)
    )

    cdf = 0.5 * (1 + jscipy.special.erf(_x / jnp.sqrt(2)))

    dist = amp * 2 * norm * cdf

    return dist


def fit_skewnorms_jax(
    x,
    *params,
):
    """
    Reproduce the behavior of `fit_skewnorms` but in JAX.
    """
    # Get the number of peaks and reshape for easy indexing
    n_peaks = int(len(params) / 4)
    params_ = jnp.reshape(jnp.asarray(params), (n_peaks, 4))
    out = 0

    # Evaluate each distribution
    for i in range(n_peaks):
        out += _compute_skewnorm_jax(x, params_[i])

    return out


def _compute_skewnorm_scipy(x, params) -> NDArray[float64]:
    amp, loc, scale, alpha = params

    _x = alpha * (x - loc) / scale

    norm = np.sqrt(2 * np.pi * scale**2) ** -1 * np.exp(
        -((x - loc) ** 2) / (2 * scale**2)
    )

    cdf = 0.5 * (1 + scipy.special.erf(_x / np.sqrt(2)))

    dist = amp * 2 * norm * cdf

    return dist


def _fit_skewnorms_scipy(x, *params):
    R"""
    Estimates the parameters of the distributions which consititute the
    peaks in the chromatogram.

    Parameters
    ----------
    x : `float`
        The time dimension of the skewnorm
    params : list of length 4 x number of peaks, [amplitude, loc, scale, alpha]
        Parameters for the shape and scale parameters of the skewnorm
        distribution. Must be provided in following order, repeating
        for each distribution.
            `amplitude` : float; > 0
                Height of the peak.
            `loc` : float; > 0
                The location parameter of the distribution.
            `scale` : float; > 0
                The scale parameter of the distribution.
            `alpha` : float; > 0
                The skew shape parater of the distribution.

    Returns
    -------
    out : `float`
        The evaluated distribution at the given time x. This is the summed
        value for all distributions modeled to construct the peak in the
        chromatogram.
    """
    if len(params) % 4 != 0:
        raise ValueError(
            "length of params must be divisible by 4\n"
            f"length of params = {len(params)}"
        )

    # Get the number of peaks and reshape for easy indexing
    n_peaks = int(len(params) / 4)
    params = np.reshape(params, (n_peaks, 4))
    out = 0

    # Evaluate each distribution
    for i in range(n_peaks):
        out += _compute_skewnorm_scipy(x, *params[i])
    return out
