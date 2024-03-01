# prepare_popt_input.params_factory

A diagram depicting the runtime flow within `params_factory` taking the peak map and windowed X and producing a table containing the lower and upper bounds and initial guess of the parameter values to fit the skewnorm distribution to X.

```mermaid
flowchart TD
pm(pm)-->window_peak_map
X_w(X_w)-->window_peak_map
window_peak_map-->wpm(wpm)
wpm-->p0_factory-->p0(p0)
p0-->bounds_factory
X_w-->bounds_factory
timestep(timestep)-->bounds_factory-->bounds(bounds)
p0-->join
bounds-->join
join-->params(params)
```

## help

[guide to symbols](https://www.researchgate.net/figure/Standard-Flowchart-Symbols_fig1_338671462)

[mermaidjs cheatsheet](https://jojozhuang.github.io/tutorial/mermaid-cheat-sheet/)

io node: `A[/Christmas/]`