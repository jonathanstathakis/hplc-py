from typing import Type, TypedDict, Callable, Any, Literal
from numpy.typing import ArrayLike
from scipy.optimize import Bounds
from scipy import sparse

from hplc import quant
class CurveFitKwargs(TypedDict):
    """
    Kwargs for scipy.optimize.curve_fit. Should be same as for the JAX implementation.

    See [the docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)

    These are the optional kwargs, excluding f, xdata, ydata, p0

    TODO: confirm that the JAX implementation uses the same kwargs
    
    NOTE: the default maximum number of iterations (dubbed 'nfev') defaults to 100 * n where n is the number of 
    variables. the original implementation of hplc_py uses a default of 1E6, and as such that is the default
    used here.
    """

    sigma: Any | None
    absolute_sigma: bool
    check_finite: bool
    bounds: tuple[ArrayLike, ArrayLike] | Bounds
    method: Literal["lm", "trf", "dogbox"]
    jac: Callable | str | None
    full_output: bool
    nan_policy: Literal["raise", "omit", None]
    least_sq_kwargs: dict[str, Any]


curve_fit_kwargs_defaults = CurveFitKwargs(
    sigma=None,
    absolute_sigma=False,
    bounds=None,
    method="lm",
    jac=None,
    full_output=False,
    nan_policy=None,
    least_sq_kwargs={"max_nfev":1E6},
)


# TODO: implement this typeddict and the defaults. atm not 100% clear on the defaults.

class LeastSquaresKwargs(TypedDict):
    """
    Optional kwargs for `scipy.optimize.least_squares`, [see docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares).

    Doesnt include `fun`, `x0`, or `bounds`, as this implementation uses bounds.
    """

    jac: Literal["2-point", "3-point", "cs"] | Callable
    methods: Literal["trf", "dogbox", "lm"]
    ftol: float | None
    xtol: float | None
    gtol: float | None
    x_scale: ArrayLike | Literal["jac"] | None
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] | Callable
    f_scale: float | None
    max_nfev: None | int  # max number of iterations, for 'trf' defined as 100 * n
    diff_step: None | ArrayLike
    tr_solver: None | Literal["exact", "lsmr"]
    tr_options: dict
    jac_sparsity: None | ArrayLike | sparse.spmatrix
    verbose: Literal[0, 1, 2]
    fun_jac_args: tuple
    fun_jac_kwargs: dict


least_squares_defaults = LeastSquaresKwargs(
    jac="2-point",
    method="trf",  # curve_fit defaults to 'trf' if bounds are provided see [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
    ftol=1E-8,
    gtol = 1E-8,
    xscale = None,
    f_scale = 1.0,
    max_nfev = None,
    diff_step = None,
    tr_solver = None,
    tr_options = {},
    fun_jac_args={},
    fun_jac_kwargs={},
)
