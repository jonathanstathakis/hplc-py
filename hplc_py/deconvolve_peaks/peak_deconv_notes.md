2023-11-28 09:36:45

Dev notes here as good as any.

Type hinting has become both crucial, and a pain. solutions:

- use Series/DataFrames where possible to provide metadata, alignment control etc of
 related data data especially where it is expected to be dimensionally homogenous.
- Use `numpy.typing.NDArray[dtype]` to type hint numpy arrays. note it needs to be
  numpy types not built-in, i.e. `np.float64` instead of `float`.
- Use pandera `DataFrameModel` and `Series[dtype]` to type hint pandas objects. store the schemas in a seperate file then import for easy reference. This allows you to track the changes