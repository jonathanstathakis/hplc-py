# Bounds Factory

The bounds factory takes the windowed time: W_t and initial guesses: p0, and produces upper and lower bound values. Calculated as follows:

 | parameter | bound | definition                          |
|-----------|-------|----------------------------------|
| amplitude |   lb  | 10% peak maxima                  |
| amplitude |   ub  | 1000% peak maxima                |
| location  |   lb  | minimum time index of the window |
| location  |   ub  | maximum time index of the window |
| width     |   lb  | magnitude of the timestep        |
| width     |   ub  | half the width of the window     |
| skew      |   lb  | -inf                             |
| skew      |   ub  | +inf                             |

```mermaid
flowchart TD

subgraph inputs
  W_x
  timestep
  skew_lb_scalar
  skew_ub_scalar
  maxima_lb_ratio
  p0_maxima
  maxima_ub_ratio
end

subgraph find_window_bounds
  W_x-->window_bounds
end

subgraph set_bounds_maxima
  maxima_lb_ratio-->bounds_maxima_lb
  maxima_ub_ratio-->bounds_maxima_ub
  p0_maxima--->bounds_maxima_lb--->bounds_maxima
  p0_maxima-->bounds_maxima_ub-->bounds_maxima
end

subgraph set_bounds_loc
  window_bounds--->bounds_loc
end

subgraph set_bounds_width
  window_bounds-->window_bounds_ub--|1/2|-->bounds_width_ub-->bounds_width
  timestep --> bounds_width_lb-->bounds_width
end

subgraph set_bounds_skew
  skew_lb_scalar--->bounds_skew_lb--->bounds_skew
  skew_ub_scalar--->bounds_skew_ub--->bounds_skew
end

subgraph bounds_pl
  bounds_maxima-->bounds
  bounds_loc-->bounds
  bounds_width-->bounds
  bounds_skew-->bounds
end
```

