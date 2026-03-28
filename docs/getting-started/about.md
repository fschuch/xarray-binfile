# About

Xarray-binfile is an xarray backend for raw binary files. It is designed for workflows where the bytes on disk are simple and efficient, but the metadata needed to interpret them lives outside the file itself.

The first target use case was binary outputs produced in workflows around the Fortran framework [2DECOMP&FFT](https://github.com/2decomp-fft/2decomp-fft) and the CFD solver [Xcompact3d](https://github.com/xcompact3d/Incompact3d). Even so, the backend is not tied to those projects and can be adapted to any compatible raw binary naming and metadata convention.

Typical examples include:

- output from Fortran or C/C++ simulation codes
- binary dumps created with `numpy.ndarray.tofile`
- one-file-per-variable or one-file-per-timestep layouts from CFD and scientific computing pipelines

The package integrates with xarray in two directions:

- reading through `xr.open_dataset(..., engine="binfile")` and `xr.open_mfdataset(..., engine="binfile")`
- writing through the `.binary_engine.to_file(...)` accessor on `xarray.DataArray` and `xarray.Dataset`

## Reading raw binary with explicit metadata

Raw binary files do not store dimension names, coordinate values, variable names, or units. Xarray-binfile therefore expects a callable compatible with `ReadSpecsGetterProtocol` that receives a path and returns the metadata needed to build a dataset.

You should also treat raw binaries as machine-specific by default. In particular, verify endianness (little-endian vs big-endian) and numeric dtype whenever files move between compilers, platforms, or architectures.

```python
import pathlib

import xarray as xr

dataset = xr.open_mfdataset(
    paths=sorted(pathlib.Path("data").glob("*.bin")),
    engine="binfile",
    read_specs_getter=user_defined_read_specs_getter,
    chunks={"z": 32, "time": 1},
    parallel=True,
)
```

With chunking enabled, the returned arrays stay lazy and Dask-backed. That makes the same code path useful for small tutorial data and for datasets that are much larger than memory.

## Writing arrays back to `.bin`

The package also registers a `.binary_engine` accessor for `xarray.DataArray` and `xarray.Dataset`. A callable compatible with `WriteSpecsGetterProtocol` controls how each array or timestep maps to filenames and per-file slices.

```python
derived.binary_engine.to_file(
    write_specs_getter=user_defined_write_specs_getter,
    directory=pathlib.Path("output"),
)
```

This is useful when you need to feed a derived result back into an external code that already expects a raw binary layout.

```{note}
If you control the storage format, standard xarray formats such as NetCDF and Zarr are usually better defaults because they preserve metadata and are easier to share across tools.
```

## Xarray and Dask fit naturally here

Xarray-binfile focuses on translating between raw binary files and xarray objects. Once the data is open, standard xarray and Dask operations apply:

- coordinate-based indexing with `.sel(...)`
- positional indexing with `.isel(...)`
- grouped and labeled reductions such as `.mean(...)`
- lazy parallel execution with Dask until `.compute()` or `.load()` is called

The tutorials in this documentation show the backend in that larger ecosystem instead of treating it as a standalone file reader.

For broader coverage of analysis, visualization, and distributed execution, see the [xarray documentation](https://docs.xarray.dev) and the [Dask documentation](https://docs.dask.org).
