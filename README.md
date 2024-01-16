
# Dev Guide

## Best Practices

2024-01-12 15:02:25

Use Pandera schemas as frame models - gives you the column names. Can add them as components. Have to add the class definition as an object rather than an instance. Then we can use the internal class object for schema checks and the global for type annotation.

2024-01-16 15:33:39

Use __init__ constants to define schema frame column aliases for consistant naming.

2024-01-16 15:34:22

The relationship between modules, files, test modules and test files should be restricted to one submodule per file, one test module per test file corresponding to one module. This is because pytest-cov coverage reports are restricted to a file granularity, but you can specify down to an individual test. Thus you cannot restrict coverage profiling to for example a class the test relates to.