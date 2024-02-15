
# Dev Guide

## Best Practices

### Package Design

1. 2024-01-24 12:52:43 - The package will consist of the pipeline nodes represented as independent submodules accepting a data structure (dataframe, but potentially a dict) and access keys to the stored 1D data seriestime, amp, etc. Those submodules will not interface with each other and will consist of pure functions, excluding their top level function which will act as the sole interface with the top level scope. Thus a heirarchy is formed, consisting of a top level user API (the Controller) containing the pipeline, and hidden submodules possessing the actions of the pipeline. The submodule data structure schemas are predefined and only change through an alias interface on entry to the submodule scope. An explicit rather than implicit schema.

2. 2024-01-30 12:24:20 - the chromatogram will be represented by the Chromatogram object, which will contain a `data` attribute, a pandas DataFrame with an x and y column. It will be initialized by passing 2 numpy arrays to it, the same length, either float or int datatype. It will perform validation within `__post_init__`. It will perform and store the timestep calculation, which can be accessed by submodules.

3. 2024-01-30 12:26:11 - The pipeline will be represented as as HPLCPY object. The pipeline will via composition apply the submodules to it, adding to the chromatogram object data as we go. Flags on the chromatogram will indicate its status.

4. 2024-01-30 14:22:38 - I need to extract knowledge out of the submodules and move them towards a sklearn estimator/transformer format. Schemas should belong to the pipeline/chromatogram (probably pipeline), the submodules should simply transform 1D arrays or return tables. For example, baseline correction returns the corrected array + optional background. Windowing returns window type and window idx arrays, deconvolution returns optimal individual peak signals and the reconstructed signal. Fit assessment returns the scores table and report card.

### DataFrame Schemas

1. 2024-01-12 15:02:25 - Use Pandera schemas as frame models - gives you the column names. Can add them as components. Have to add the class definition as an object rather than an instance. Then we can use the internal class object for schema checks and the global for type annotation.


2. 2024-01-16 15:33:39 - Use __init__ constants to define schema frame column aliases for consistant naming.

3. 2024-01-24 12:20:05 - Building on 1., You can access the column names and column ordering from the schema, and probably the data types too.

### DataFrame Table Formats

1. It is difficult to define a storage format for transformations of a signal. The best method I can see now is to store everything in long form with a set number of static columns, or fields, relating to each physical property of the phenomenon + labels to identify each stage of the transformation. i.e. for the signal, setup a signal table, then add a baseline corrected version underneath within the same fields, labelled 'bcorr', then for the reconstruction, use 'bcorr+recon' as the label, etc. Thus 1 schema is necessary, with x 'isin' checks on the label column. This way, you pivot when necessary, but do not need to mutate the schema to add property identical but value different columns.

### Testing

1. 2024-01-16 15:34:22 - The relationship between modules, files, test modules and test files should be restricted to one submodule per file, one test module per test file corresponding to one module. This is because pytest-cov coverage reports are restricted to a file granularity, but you can specify down to an individual test. Thus you cannot restrict coverage profiling to for example a class the test relates to.

2. 2024-01-31 10:01:55 - To automate dataset specific schema creation, I need to be able to read the current schema file, generate the schema to add, add the schema, and modify in place if necessary. so i need an IO module, a "sense" module and a generation module. The sense module will contain the rules within the schema file. the rules include: seperation of the file into schema 'regions' bound by lines, detection of schema, parsing of schema. Once the schema are generated I can then use them for type annotation. You could produce a diff module as well to warn if there is a diff between

3. 2024-02-05 19:52:58 - I cant run the tests on my datasets if there are 'compare main' tests littered throughout parametrizable tests as there is no guarentee that the main version will be able to decompose my data. The quickest solution will to make a 'compare main' test dir which contains those tests.

4. 2024-02-07 11:49:30 - I have stored the expected window bounds of the asschrom dataset at "/Users/jonathan/hplc-py/tests/tests_jonathan/test_data/asschrom_window_bounds.csv". The reason being that we need sanity checks - on developing the pipeline for use with MY data, one peak has been assigned to interpeak when joining with the windowed signal. Checking the asschrom data against that csv file will confirm whether the pipeline is working as expected.

### Visualisation

1. 2024-01-30 11:46:35 - Use matplotlib. It gives you the capability to create very customised plots with a high level of control compare to other packages.

2. 2024-01-30 11:47:17 - Matplotlib does not allow you to construct visualisations incrementally, rather any drawings intended to be combined together need to be drawn on the same object. Thus visualisation functions need to be designed to accept an `ax` object to draw on, and return that `ax` object for downstream drawing.

3. 2024-02-08 00:53:47 - Following on point 2... The problem with matplotlib is that their API is unpythonic, relying heavily on get and set functions. To use it, you have to understand that everything drawn is either a 'line2d' object, which includes markers, or a patch, which is a 2D demarcated space. These are known as artists. Axes contain artists, and the artists accessors are used to reveal them. All the properties, from the data to marker edge width, can then be viewed. As google treats time as a flat surface, and as matplotlib has changed considerably over the years, the best method of obtaining up to date information is through the help function.

### Forking Third Party Libraries

1. 2024-02-05 13:06:16: This project makes extensive use of Pandera for function IO validation, datatype coercion, and parametrization of dataframe accessors. Unfortunately a lot of these features are not fully fleshed out or buggy. These include the error messages, and return types of attributes such as column names. I have raised several issues with the maintainers (owners) of the repo, but have received no response. My next approach will be to fork Pandera and begin making my own modifications. The problem with this is - how do I keep the fork up to date with releases from main? The answer appears to be [rebasing](https://www.reddit.com/r/git/comments/z01ejf/keeping_a_fork_updated_best_practices/). According to the commenters here, rebasing makes it so my local modifications are based on the up to date version of main rather than the version I branched. Notes on rebase: [3.6 Git Branching - Rebasing](https://git-scm.com/book/en/v2/Git-Branching-Rebasing). 

The second question is this: how do I use the fork in my project? The answer is to set up a local repo somewhere then install in editable mode (or not) within the project. Best practice would be to name the forked library something else to distinguish it from the main project.

### Interfaces

1. 2024-02-06 13:13:12: The submodules of the pipeline operate in different domains, do different things, and rely on each other. But they also need to be independent. For each class, there should be a pattern of fitting and transforming the input, then storing the input, intermediates and output as instance attributes for later inspection. Thus a pipeline object would merely act as a mediary, telling an internal chromatogram object to acquire the transform output before moving on. Thus, the baseline correction module will store the corrected as 'corrected', background as 'background', etc. map_peaks will store the peak_map as 'peak_map', map_windows will store the windowed_signal as 'windowed_signal', deconvolution will store the reconstructed signal as 'reconstruction'. To emulate scikit learn, we will convert all to a fit, transform format.

2. 2024-02-06 13:52:37: based loosely on the sci-kit learn API, data agnostic settings are psased to __init__, data AND specific settings are passed to fit, then tranform is called to do the action.
3. 2024-02-06 14:02:16: Using pandera schemas to share information across modules was a terrible idea because you lose encapsulation, and encapsulation is everything right now. Get rid of it wherever it can be found. Can still use schemas for io validation, but you cant refer to them for dataframe info.

4. 2024-02-09 12:25:42: To clarify which are front and which are back end classes, front ends will be prefixed with "UI_". This is merely a precaution during development in order to encourage a heirarchy.