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

Q2: can we tweak window two in such a manner that the peak width is better estimated? The width should be somewhere between 