2023-11-28 09:36:45

Dev notes here as good as any.

Type hinting has become both crucial, and a pain. solutions:


- Use `numpy.typing.NDArray[dtype]` to type hint numpy arrays. note it needs to be
  numpy types not built-in, i.e. `np.float64` instead of `float`.
- Use pandera `DataFrameModel` and `Series[dtype]` to type hint pandas objects. store the schemas in a seperate file then import for easy reference. This allows you to track the changes

Design Spec:
- establish an explicit core dataframe (database)
- satellite modules use numpy arrays
- Thus Chromatogram and 'fit_peaks' has operates on a df acting as an interface between the df and the satellite modules.
