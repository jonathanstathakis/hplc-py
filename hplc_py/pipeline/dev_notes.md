2024-03-31 04:17:13

Another logbook.

# Pipelines as Classes

Debugging pipelines is difficult. This is because they can have many levels of operation with many dependencies, involve complex tasks, and handle one or more simple or complex inputs, and sometimes (usually) it is hard to define what erronous output (or input) data looks like, especially during development. Also, a fundamental fact of the pipeline data structure is a linear (?), ordered dependency, thus the failure point may be further up the line than where the actual error is detected. Furthermore, as pipelines grow in complexity, the overview position becomes more and more the base state of the developer. Finally, visualisations are more useful than raw numbers for observing the 'health' of a pipeline, as pattern memory serves a key role in how humans percieve data. Thus the visualisation should be a first-class interface to the pipeline, and a method of creating visualisations locally but handing them off to the overwatch/overview for observation on error is key.

Thus is something that a functional approach to the pipelines fails at, because if the visualisation is created locally, then the viz object needs to be passed through the call stack to the overwatch, complicating function returns unnecessarily. A class approach would enable overwatch to access a local viz method and enable callbacks if necessary. An unbound level of information about the function of the pipeline could be stored and accessed only as necessary, hiding clutter. This of course would be equivalent to defining global data stores relative to the functions, which would be an approach from a purely functional paradigm (?).

Anyway, its a quick modification. An interface needs to be defined, i.e. fit, transform, and then any plot functions need to be defined internally and are to return the plot object. Tables likewise, and potentially a formatted report.

For example, `pipe_preprocess_data` could initialise with which transformations to use (i.e. "correct_baseline") and their kwargs, have a `fit` function which takes the `data`, `key_time`, and `key_amp`, and then call a `transform` function, which would return the output (the overall API is base on [scikit-learn](https://scikit-learn.org/stable/developers/develop.html) with downtrack goal of full scikit learn integration).

2024-03-31 06:31:20

That has worked exceptionally well. Defining the external interface occludes the chaos below, and enables the pure pipeline flow of data to be seperated from dev and debugging functionality. During development, it was found to be logical to wrap viz into report, and to begin to consider defining a formal structure for report through yet another abstract base class - a Report type. it would have to be simplistic to be usable throughout the pipeline, but would enable clear use of the report downpipe. For example, it could contain a table property, obviosuly containing relevant tabular data, and a viz property, containing a viz. They could then be whatever is best for the specific Pipeline. for example the table property could contain an intermediate values table and a scores table, or aggregations.

