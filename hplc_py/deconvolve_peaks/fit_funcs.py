"""
Curve fit functions to be implemented in PeakDeconvolver.popt_factory
"""

def optimizer_jax(
        self,
    ):
        from jaxfit import CurveFit

        cf = CurveFit()
        return cf.curve_fit
    

def optimizer_scipy(
        self,
    ):
        from scipy.optimize import curve_fit

        return curve_fit