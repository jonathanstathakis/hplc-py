2024-03-04 11:53:43

Debugging deconvolution for the 15th time.

Comparing the inputs and outputs of popt reveals no significant differences bar the upper bound scale of peak 3, which is ~59 to clabs 7. No clear reason why so will need to investigate that first.

2024-03-04 12:15:32

The scale ub calculation was wrong, used the division of the window end time value rather than the division of the length of each window. fixed, but no change to popt differences.

Next point to check will be the deconvolution input. The input time series has been a problem before, could be again.

Looks right to me..

can we test it with the cremerlab params?

Fixed it. Was trying to deconvolve based on X_idx units rather than X.  The internal references to definitions makes it hard to debug due to loss of encapsulation.

2024-03-06 11:53:00

Trying to design the progress bar. What we currently have is an algorithm that takes a max_nfev and divides it into subsets. This therefore imposes the requirement that each subset gets enough nfev, which is an error we have to be able to handle. This also means that the optimization keeps running until max_nfev is reached, rather than terminating as expected when a solution is found. So thats two things to do:

- handle the case of an error during optimization
- tell it to stop when optimization is reached through successful termination condition.

All of this was based on the idea that I could collect intermediate results. I dont think thats true. if it errors out when it reaches input max_nfev, how can I collect the results?

What does scipy curve_fit actually do? because it looks like it masks a lot of information returned by least_squares.

Ok so passing verbose to least_squares will allow us to print a progress report during iterations. just add the kwargs options, least squares has a 'max_nfev' option that should do what we expect. It does, without error. Is curve fit throwing the error? yes it is. why the fuck did they write it like that. What if we recreate curve fit without it..
 
So the error occurs in all cases when least_squares result attribute `success` is False. no way around that.

again, what does curve fit provide over a direct least_squares call?

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

Thats it. So a bunch of input validation and edge case handling which we could reimplement without that rediculous error raising..

Considering that I dont care about any of it and my input is already validated i can just drop curve fit in favor of least squares..

Now we've got the elapsed times, and we can get all the information from least_squares while running partial fits. To implement the progress bar we need to again base it off of the max_nfev submitted, and implement a loop that continues until the success message is passed. What does that mean? what that means is that every window is successfully fitted. How do we report that? each window is fitted the same number of times, treated independently. so one progress bar for windows, another for the intermediate fits.

Lets be explicit about the call heirarchy in deconvolution:


1. deconvolution_pipeline
2. curve_fit_windows
3. popt_factory
4. get_popt
  - iterates over the max_nfev divisions.
5. curve_fit_
6. least_squares

2024-03-07 00:19:21

Time to assemble my analysis tool.

At a basic level it just needs to provide the fit report.