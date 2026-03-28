# Further Reading

This section points to the main libraries and references that complement xarray-binfile in real workflows. The package is intentionally small: it focuses on mapping raw binary files into xarray objects and writing data back to the same kind of layout. Most of the higher-level analysis, visualization, and scaling features come from the surrounding scientific Python ecosystem.

## Xarray

- [Xarray documentation](https://docs.xarray.dev/)
- [Xarray user guide](https://docs.xarray.dev/en/stable/user-guide/index.html)
- [Xarray I/O guide](https://docs.xarray.dev/en/stable/user-guide/io.html)
- [Xarray plotting guide](https://docs.xarray.dev/en/stable/user-guide/plotting.html)

## Dask

- [Dask documentation](https://docs.dask.org/)
- [Dask array documentation](https://docs.dask.org/en/stable/array.html)
- [Dask diagnostics and visualization](https://docs.dask.org/en/stable/diagnostics-local.html)
- [Why Dask works well with xarray](https://docs.xarray.dev/en/stable/user-guide/dask.html)

## NumPy and raw binary I/O

- [`numpy.fromfile`](https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html)
- [`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [`numpy.ndarray.tofile`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html)

## Storage formats and portability

- [NetCDF support in xarray](https://docs.xarray.dev/en/stable/user-guide/io.html#netcdf)
- [Zarr support in xarray](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr)
- [Zarr specification](https://zarr.readthedocs.io/)

## Domain workflows

- [Xcompact3d](https://xcompact3d.readthedocs.io/)
- [2DECOMP&FFT](https://2decomp.org/)
- [Python Packaging User Guide](https://packaging.python.org/)
