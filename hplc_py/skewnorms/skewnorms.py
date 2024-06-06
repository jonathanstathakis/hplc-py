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
    """
    alpha is the skew parameter

    $ (x - loc ) / scale $ is a term used to shift the location of the distribution.
    
    eg: dist.pdf(x, loc, scale) = standard_dist.pdf((x - loc)/scale) / scale
    """
    amp, loc, scale, alpha = params

    x_minus_loc = x - loc

    _x = alpha * (x_minus_loc / scale)

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


