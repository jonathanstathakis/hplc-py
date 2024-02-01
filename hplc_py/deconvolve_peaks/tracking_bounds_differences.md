2024-01-31 00:02:05

This document will track the changes brought by modifying the default lb of width from timestep to 1% of the peak whh.

The current average runtime of the pipeline up to the end of the deconvolution is 2.8 seconds, with ~ 1.30 seconds per peak window.

I need to track whether the output changes, which it will, but the question is by how much.

To track, the easiest thing to do is probably to copy paste the code, import them into the same test and compare, that way I can compare them at run time.

2024-02-01 10:02:56

Got massively out of scope. started developing a promising program that was going to take me hours. ran into 
issues with parsing the datatypes. Started playing with `pa.infer_schema` and found that
the `to_script` method generated basic upper and lower bound checks for numerics, which
I consider good enough to continue. End of the day the fit assessment will be the final
indicator of a problem, and I should be flexible with the actual values.

To implement the  `infer_schema.to_yaml` I need a class that is able to be added to
a test to create the schema if none exists, otherwise read from existing schema.

