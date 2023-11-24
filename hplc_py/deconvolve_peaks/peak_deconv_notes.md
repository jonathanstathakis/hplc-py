2023-11-22 13:18:32

Continuing on from XGBoost notes. Have heavily refactored the hplc-py package for clarity of function and debugging. But still getting the same infeasible error. Time to build a bug report.

I need to extract the state of the window that is causing the error. That means:

current window index
current peak index
current bounds
current guess

This should be a class.

2023-11-22 15:06:32

Report class 'WindowState' done and operational.

From the report we can see that the second window 4th peak is causing the problem. Can we also generate a plot of the signal?

Thats done from window df, but could also be done from window report..

2023-11-22 15:29:12 - observing peak 4 of window 2, there is no obvious cause for the erronous width allocation..

2023-11-22 19:12:30 I have generated a joined table with all information relevant to a peak and peak window for output. I have also produced a plot display of the id'd peaks in a window, and the estimated widths, in order to gauge how its getting it wrong.

From this we can see that peak 2,2 (window two, peak two) is almost completely subsumed by peak 2,1

2023-11-23 00:12:42 - the window in question is ~1.9 to ~4.9.

first question:

Q1: what happens when we run window 1 only?

Q2: can we tweak window two in such a manner that the peak width is better estimated? The width should be somewhere between..

2023-11-23 23:58:35

There is an abundance of information about the fundamentals of peak deconvolution, namely [this](http://emilygraceripka.com/blog/16) post from Dr. Ripka. There is also a series of detailed tutorials by the University of Maryland [here](https://terpconnect.umd.edu/~toh/spectrum/SignalsAndNoise.html).

From this and the `hplc-py` code we can deduct that `scipy.signal.peak_widths` is the problem, it is underestimating the width of peak (2,2). Sharpening the peak may increase the measured width enough to get it over the default bound. Furthermore, following Ripkas example we could create our own fitting regime, at least to understand the problem.

First thing i need to do is to extract the signal then apply a sharpening filter ( which it appears I will have to define), then experiment with `peak_widths`.

The window is from 2 to 4.5.

2023-11-24 08:59:17

so sharpening the peaks is resulting in a pretty good fit for the majority of the peaks, however we're encountering an undefined error in the first 2 min region, and some of the seriously overlapped peaks in the later areas are not well fit because not all of the component peaks are being detected so the peak that is is being modelled as very skewed. Need to find a compromise. Got all the moving parts here, just need the right combination.

~~2023-11-24 10:02:51 I am trying to produce a subplot of `chm.show()` cut into bins to display different regions. To do this I have had to modify the function to accept a fig and ax object to plot on, and now I have encountered an interesting issue - the output of `fit_peaks` is not de-normalized. Im not sure if thats by design or not, but we should add it in. Best way to do that will be to define the inverse of the normalization function.~~

2023-11-24 12:07:44 - the answer is that the peak df is not peak properties, it is the best fit values. In the case of amplitude it appears that even though the documentation states that it is the amplitude maxima, it appears to be the amplitude of the centroid. I suppose the authors didnt think that the true amplitude was worth reporting when the area is of more interest. How can I get the peak amplitudes? From the reconstructed signals.

2023-11-24 12:38:19 - have added peak maxima to `peak_df` and stored `unmixed_chromatograms` as a DF with the peak id as columns and retention time as index.

2023-11-24 13:11:22 - have pushed all recent modifications to remote. Now to look into optimizing the fit. We've still got the problem with 0 - 2 mins to solve as well..

I want to start over with the raw data prior to resampling and see if that has any effect on the results. To do that best thing will be to install my branch in my main project.