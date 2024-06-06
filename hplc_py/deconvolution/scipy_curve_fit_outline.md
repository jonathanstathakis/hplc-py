# Description of `curve_fit` Body

1. 862: validates p0 by checking dimensionality
2. 863: sets n to `p0.size`
3. 868: sets lb and ub through `prepare_bounds`
4. 872-877: assigns method by checking if bounded problem or not.
5. 888: casts ydata to float64
6. 898: casts xdata to np array
7. 900: validates that ydata is not empty
8. 905: - 923: NAN handling.
9. 925: - 943: handle sigma
10. 947: - 950: handling callable jac
11. 952: - 956: handling func args signature
12. 975: least squares call
13. 978: error if sucessful fit not achieved
14. 985 - 987: set ysize to len res.fun, cost to 2 * res.cost, popt to res.x
15. 989 - 994: compute pcov through Moore-Penrose inverse of jac while getting rid of zero singular values
16. 997 - 1012: handle pcov edge cases, if pcov is none or has NA, generate a pcov with dimensions the length of popt filled with inf. if not absolute sigma, and ysize > p0.size calculate pcov as pcov * sum of squares, otherwise replaces pcov values with inf. All of these optinos rase a warning that covariance of parameters cannot be estimated.