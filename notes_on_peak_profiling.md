2023-12-05 12:36:43

Investigate findwindows settings to produce an expected window range, that is:
- isolated peaks are covered by the window from left base to right base + buffer
- overlapping signals are considered in the same window, again, from left to right base
of the *region* + buffer.

TODO:
- [ ] in `baseline` rename the peak_df 'amp' column to 'amp_input', 'norm_amp' to 'amp_norm'

# Peak Prominence

Peak prominence is defined as the difference of the peak maxima and its lowest contour line.
 
Peak prominence calculation method:
1. Define window interval
    - extend a a horizontal line left or right of the peak maxima.
    - the extension stops either at a window bound ('wlen') or when the line encounters the slope again.
2. Define signal left and right bases.
    - find signal minima for the left and right window bound.
3. Calculate prominence
    - the higher of the left or right base is defined as the lowest contour line of the peak.
    - prominence is calculated as the difference between the peak maxima and its lowest contour line.
    
# Peak Widths

1. Evaluation height is calculated as peak maxima - peak prominence * rel_height. The more prominent the peak, the lower down the eval height.
2. draw a line at the evaluation height in both directions until:
    - the line intersects a slope
    - signal border
    - crosses the vertical position of the base
3. width is calculated as the distance between the endpoints defined in (2.). By definition the maximum width is the horizontal distance between the bases.

Currently, the width and the