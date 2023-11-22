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

Generated from 