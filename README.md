
# Dev Guide

## Best Practices

### Package Design

1. 2024-01-24 12:52:43 - The package will consist of the pipeline nodes represented as independent submodules accepting a data structure (dataframe, but potentially a dict) and access keys to the stored 1D data seriestime, amp, etc. Those submodules will not interface with each other and will consist of pure functions, excluding their top level function which will act as the sole interface with the top level scope. Thus a heirarchy is formed, consisting of a top level user API (the Controller) containing the pipeline, and hidden submodules possessing the actions of the pipeline. The submodule data structure schemas are predefined and only change through an alias interface on entry to the submodule scope. An explicit rather than implicit schema.

2. 2024-01-30 12:24:20 - the chromatogram will be represented by the Chromatogram object, which will contain a `data` attribute, a pandas DataFrame with an x and y column. It will be initialized by passing 2 numpy arrays to it, the same length, either float or int datatype. It will perform validation within `__post_init__`. It will perform and store the timestep calculation, which can be accessed by submodules.

3. 2024-01-30 12:26:11 - The pipeline will be represented as as HPLCPY object. The pipeline will via composition apply the submodules to it, adding to the chromatogram object data as we go. Flags on the chromatogram will indicate its status.

4. 2024-01-30 14:22:38 - I need to extract knowledge out of the submodules and move them towards a sklearn estimator/transformer format. Schemas should belong to the pipeline/chromatogram (probably pipeline), the submodules should simply transform 1D arrays or return tables. For example, baseline correction returns the corrected array + optional background. Windowing returns window type and window idx arrays, deconvolution returns optional individual peak signals and the reconstructed signal. Fit assessment returns the scores table and report card.

### DataFrame Schemas

1. 2024-01-12 15:02:25 - Use Pandera schemas as frame models - gives you the column names. Can add them as components. Have to add the class definition as an object rather than an instance. Then we can use the internal class object for schema checks and the global for type annotation.



2. 2024-01-16 15:33:39 - Use __init__ constants to define schema frame column aliases for consistant naming.

3. 2024-01-24 12:20:05 - Building on 1., You can access the column names and column ordering from the schema, and probably the data types too.

### DataFrame Table Formats

1. It is difficult to define a storage format for transformations of a signal. The best method I can see now is to store everything in long form with a set number of static columns, or fields, relating to each physical property of the phenomenon + labels to identify each stage of the transformation. i.e. for the signal, setup a signal table, then add a baseline corrected version underneath within the same fields, labelled 'bcorr', then for the reconstruction, use 'bcorr+recon' as the label, etc. Thus 1 schema is necessary, with x 'isin' checks on the label column. This way, you pivot when necessary, but do not need to mutate the schema to add property identical but value different columns.

### Testing

2024-01-16 15:34:22

The relationship between modules, files, test modules and test files should be restricted to one submodule per file, one test module per test file corresponding to one module. This is because pytest-cov coverage reports are restricted to a file granularity, but you can specify down to an individual test. Thus you cannot restrict coverage profiling to for example a class the test relates to.

### Visualisation

1. 2024-01-30 11:46:35 - Use matplotlib. It gives you the capability to create very customised plots with a high level of control compare to other packages.
2. 2024-01-30 11:47:17 - Matplotlib does not allow you to construct visualisations incrementally, rather any drawings intended to be combined together need to be drawn on the same object. Thus visualisation functions need to be designed to accept an `ax` object to draw on, and return that `ax` object for downstream drawing.