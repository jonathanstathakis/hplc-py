TODO:

- [ ] write 'stored_popt' for both datasets- currently only testing on asschrom

TODO:

- [x] plot whh to understand what it is.
- [x] test asschrom dataset with original code to see what the width values should be
- [x] diagnose why my code produes a different value
    - reason 1 is that the peak widths are stored in time units rather than index units.
    - thats the only reason. same same, dont change a thing.
- [x] write up on the skew norm model to understand:
    - [x] why parameter choices are what they are
    - [ ] interpretability of the results, how reliable are they?
    - [ ] what are the danger points of the skewnorm model
        - the last two are currently unanswered.
- [x] adapt .show
    - [x] plot raw chromatogram
    - [x] plot inferred mixture (sum of reconstructed signals)
    - [x] plot mapped peaks (fill between)
    - [ ] add custom peaks subroutine
- [x] identify why the fitted peaks do not match the original signal.
    - [x] define a fixture class that returns the parameters of the hplc_py calculation for the same dataset. Provide an interface that outputs the parameters in the same format as your definitions for easy comparison.
        - [x] output the results from the main env to a parquet file
        - [x] write the fixture to read the output
        - [x] write tests to compare my results with theirs.
        - [x] add a params_df output to the fixture, this being the lb, p0, ub in similar format to mine.
        - [x] add timestep to params_df
    - [x] write an adapter (proto decorator) to express the main calculated parameters as your format to feed to `_popt_factory` directly.
    - [x] 2023-12-17 09:35:43 - get back to a functioning test base. solve the test problems* 
    - [ ] add schemas for the main AssChrom dataset at each stage of the process
        - [ ] find windows
        - [ ] deconvolution
        - Note: this will be difficult because we'd have to adapt at every stage, manually recording the data and reformatting it. Not impossible.
    - [x] determine why your p0 values are rounded to 3 decimal places. Answer: erronous type casting to int after the width calculations. Solution: casting all width measurements to float instead.
    - [x] determine why the amp values deviate at the third decimal place. Hypothesis: baseline corrected signals differ. How to test:
        - [x] output main signal, baseline corrected signal, background
        - [x] isolate my corresponding series
        - [x] compare
        - outcome of comparison - all values are equal until the optimization.
        - outcome: baseline corrected signals differ by an order of magnitude of 2 to 3. Need to investigate why this is happening. This will be achieved by first creating intermediate variable tables to compare the values
        - [x] compare the debug dfs
    - [x] normalization only needs to occur during the peak profiling. we dont refer to it afterwards, so move it to that point.
        - [x] need to define and apply a normalize inversion function to convert the peak width measure height calculations to base scale.
    - [ ] determine why WHH varies by a small magnitude.
- [ ] parametrize all module inputs to enable higher level control of variable flow
- [ ] enforce a resampling during data loading, then use timestep calls to convert to time units rather than joining
    
            
- [ ] adapt fit assessment module(s)
    - [ ] score_df
      - [ ] define score df factory
      - [ ] define interpeak windows
      - [ ] test score df against peak and interpeak windows
      - [ ] compare results with main

- [ ] adapt map_peaks (?)
- [ ] add in the customization routines
- [ ] build the master method
- [ ] seperate initial peak profiling from window finding 
- [ ] make the modules more abstracted, expose intermediate values as class objects.

