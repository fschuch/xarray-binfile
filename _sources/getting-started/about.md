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

The read engine is registered automatically with xarray as a plugin entry point. The write accessors become available as soon as `xarray_binfile` (or any of its modules) is imported.

Xarray-binfile focuses on translating between raw binary files and xarray objects. Once the data is open, standard xarray and Dask operations apply:

- coordinate-based indexing with `.sel(...)`
- positional indexing with `.isel(...)`
- grouped and labeled reductions such as `.mean(...)`
- lazy loading and parallel execution with Dask until `.compute()` or `.load()` is called

The tutorials in this documentation show the backend in that larger ecosystem instead of treating it as a standalone file reader.

For broader coverage of analysis, visualization, and distributed execution, see the [xarray documentation](https://docs.xarray.dev) and the [Dask documentation](https://docs.dask.org).
