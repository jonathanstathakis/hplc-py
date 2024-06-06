"""
2024-04-08 12:14:00

An attempt to define polars expressions for finite difference calculations. Became too 
much work because I would have to verify the code, and there were varying (read unclear)
definitions on how delta x was defined between the different 'differences'. Have opted
for using `pl.diff`, i.e. forward diff, for initial calculations.

# Notes

It is not wise to proceed beyond the first derivative as the finite difference is VERY sensitive to noise, and error in general, see: <https://services.math.duke.edu/~jtwong/math361-2019/lectures/Lec6Differentiation.pdf>. An example in the wild of foreward diff noise can be seen here: <https://stackoverflow.com/questions/69000410/extract-and-plot-the-first-derivative-of-a-curve-in-python-without-knowing-its-f>

Better to curve fit then derive that.
"""

import polars as pl


class FiniteDifferenceExpressions:
    """
    Approximate the first and second derivatives through the finite difference <https://en.wikipedia.org/wiki/Finite_difference>.

    <https://terpconnect.umd.edu/~toh/spectrum/Differentiation.html>

    # Finite Difference

    ## What is the Finite Difference

    The derivative is the rate of change of a FUNCTION, however most scientific exercises deal with data rather than known functions. Thus the derivative needs to be approximated from the data. One method is through the finite difference.

    It is named 'finite difference' because the definition does not attempt to find the limit as h approaches zero <https://en.wikipedia.org/wiki/Finite_difference_method>

    - Use forward difference at the start point, back at the end point, central for the interior points <https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf>

    - why central difference is the best approximation <https://math.stackexchange.com/questions/4267404/intuitively-why-does-the-central-difference-method-provide-us-a-more-accurate-va>
    answer: its the average between the foreward and back differences (if time is equally spaced)
    - the calculation of the 2 point central difference forgoes y value reference to the current datapoint, rather is the difference between the future data point an the past data point for the CURRENT data point. See <https://dmpeli.math.mcmaster.ca/Matlab/Math4Q3/NumMethods/Lecture3-1.html>

    Calculation:
    I'(t_0)=(I_{1} - I_{-1})/(t_{1}-t_{-1})

    ## Second derivative approximation

    given as:

    I'(t_0)=(I_{1} - 2I_{0} + I_{-1})/(t_{1}-t_{-1})^2

    See <https://terpconnect.umd.edu/~toh/spectrum/Differentiation.html>

    Sources:
    good for notation - <https://services.math.duke.edu/~jtwong/math361-2019/lectures/Lec6Differentiation.pdf>
        Notes: due to the formulation, as h approaches zero, error approaches infinity

    """

    def __init__(self, x_key: str, y_key: str):
        """
        h: difference in the x values
        f: forward relative to y
        b: back, relative to y
        """
        self.x_key: str = x_key
        self.y_key: str = y_key
        self.x: pl.Expr = pl.col(self.x_key)
        self.y: pl.Expr = pl.col(self.y_key)

        # the shifted series, f: forward, b: back.
        self.y_1f: pl.Expr = self.y.shift(-1)
        self.y_2f: pl.Expr = self.y.shift(-2)
        self.y_1b: pl.Expr = self.y.shift(1)
        self.y_2b: pl.Expr = self.y.shift(2)

        self.x_1f: pl.Expr = self.x.shift(-1)
        self.x_1b: pl.Expr = self.x.shift(1)

    def first_order_foreward_diff(self):
        """
        I'(t_0)=(I_{1}-I_{0})/(X_1-X_0)
        """
        foward_diff_first = self.y_1f.sub(self.y).truediv(self.x_1f.sub(self.x))
        return foward_diff_first

    def first_order_back_diff(self) -> pl.Expr:
        """
        I'(t_0)=(I_{1}-I_{0})/(X_1-X_0)
        """
        foward_diff_first: pl.Expr = self.y.sub(self.y_1b).truediv(
            self.x.sub(self.x_1b)
        )
        return foward_diff_first

    def first_order_central_diff(self) -> pl.Expr:
        """
        return a polars expression to calculate the central difference to approximate the derivative. Use Central diff for interior points, forward for j = 1, back for j = n.
        """

        central_diff_first = self.y_1f.sub(self.y_1b).truediv(self.x_1f.sub(self.x_1b))
        return central_diff_first

    def second_order_forward_diff(self) -> pl.Expr:
        # I"_{j} = (I_{j+2} - 2I_{j+1}+I_{j})/(h**2) <https://en.wikipedia.org/wiki/Finite_difference>

        diff: pl.Expr = (self.y_2f.sub(self.y_1f.mul(2)).add(self.y)).truediv(
            self.x.pow(2)
        )

        return diff

    def second_order_back_diff(self) -> pl.Expr:
        # I"_{j} = (I - 2I_{j-1} + I_{j-2})/(h ** 2) <https://en.wikipedia.org/wiki/Finite_difference>

        diff: pl.Expr = (self.y - pl.lit(2).mul(self.y_1b) + self.y_2b).truediv(
            self.h.pow(2)
        )
        return diff

    def second_order_central_diff(self):
        # I"(t_0)=(I_{1} - 2I_{0} + I_{-1})/(t_{1}-t_{-1})^2
        central_diff_second = (
            self.y_1f.sub(pl.lit(2).mul(self.y)).add(self.y_1b).truediv(self.h.pow(2))
        )

        return central_diff_second
