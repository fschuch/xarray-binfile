# About

Xarray-binfile is an xarray backend for raw binary files. It is designed for workflows where the bytes on disk are simple and efficient, but the metadata needed to interpret them lives outside the file itself.

The first target use case was binary outputs produced in workflows around the Fortran framework [2DECOMP&FFT](https://github.com/2decomp-fft/2decomp-fft) and the CFD solver [Xcompact3d](https://github.com/xcompact3d/Incompact3d). Even so, the backend is not tied to those projects and can be adapted to any compatible raw binary naming and metadata convention.

Typical examples include:

- output from Fortran or C/C++ simulation codes
- binary dumps created with `numpy.ndarray.tofile`
- one-file-per-variable or one-file-per-timestep layouts from CFD and scientific computing pipelines

## Why not just `numpy.fromfile`?

A call to `numpy.fromfile` returns a flat array with no shape, no dimension names, no coordinates, and the whole file resident in memory. That is fine for a single small file, but turns every multi-file dataset into a hand-written loop over paths, reshapes, and index arithmetic.

The natural next step is `numpy.memmap`. It maps the file into memory with a known shape and dtype, so slicing it reads only the bytes that are touched, and that is exactly what xarray-binfile does under the hood: a request for a whole file goes through `numpy.fromfile`, while a request for a slice goes through `numpy.memmap` and only the selected region is read. On its own, however, a memmap is still a single unlabeled array for a single file. The caller still has to know which file holds which variable and time step, and which integer offsets correspond to which physical location.

Xarray-binfile does not try to improve on NumPy as a byte reader. Its value is that the metadata you provide is enough to hand the files to xarray, and from there everything is handled by xarray, Dask, and the tools built around them:

- **One labeled dataset from many files.** The file-name convention is folded into coordinates, so a slab at a given time or height is selected by value rather than by computed offsets. Alignment, broadcasting, `groupby`, `rolling`, `coarsen`, and reductions along named dimensions all work out of the box.
- **Lazy loading and larger-than-memory data.** Opening a dataset reads no bytes until a value is needed, and only the chunks required by a computation are ever read. See [Parallel and larger-than-memory](../tutorials/parallel-and-larger-than-memory.ipynb).
- **Parallel execution.** The same code runs on a laptop with the threaded scheduler or on a cluster with [Dask Distributed](https://distributed.dask.org), with no changes to how the files are opened.
- **Visualization.** `DataArray.plot` builds matplotlib figures with axes labeled from the coordinates, and [hvplot](https://hvplot.holoviz.org) or [xarray's plotting integrations](https://docs.xarray.dev/en/stable/user-guide/plotting.html) add interactive views. See the [examples](../tutorials/examples.ipynb).
- **Conversion to portable formats.** A dataset can be written to [netCDF](https://docs.xarray.dev/en/stable/user-guide/io.html#netcdf) with `to_netcdf` or to [Zarr](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr) with `to_zarr`. Both carry dimension names, coordinates, and attributes with the data, so the metadata convention no longer has to travel separately. Zarr in particular keeps chunked, lazy access while being readable from object storage. Both conversions are shown in [Write](../tutorials/write.ipynb).
- **Interoperability.** `to_dataframe` hands data to pandas, `.values` exposes NumPy arrays for SciPy and friends, and on a chunked dataset `.data` exposes the underlying Dask array for custom graphs. Derived quantities keep their coordinates, so results can be written back to any of the formats above. The [examples](../tutorials/examples.ipynb) include exporting probe time series to CSV this way.

None of these are available in pure NumPy without writing them yourself. The tutorials in this documentation show the backend in that larger ecosystem instead of treating it as a standalone file reader.

The package integrates with xarray in two directions:

- reading through `xr.open_dataset(..., engine="binfile")` and `xr.open_mfdataset(..., engine="binfile")`
- writing through the `.binary_engine.to_file(...)` accessor on `xarray.DataArray` and `xarray.Dataset`

The read engine is registered automatically with xarray as a plugin entry point. The write accessors become available as soon as `xarray_binfile` (or any of its modules) is imported.

## Origin

Xarray-binfile started life inside [xcompact3d-toolbox](https://github.com/fschuch/xcompact3d-toolbox), the post-processing toolbox for Xcompact3d. There it began as a thin layer around `numpy.fromfile`: each call read a whole file into memory and wrapped the result in an xarray object, with some extra syntactic sugar on top. A dict-like interface made it possible to select the entire time series of a given variable, or every variable from a given snapshot, without writing the file-name bookkeeping by hand.

Moving from "read one file at a time" to "open every file at once and load from disk only on demand" called for a real xarray backend, so the code was moved into a stand-alone package. The hope is that it can serve a wide range of raw binary conventions, not just Xcompact3d. The toolbox now depends on xarray-binfile for its file I/O. Credit is also due to the xarray community for the guidance given in [this discussion](https://github.com/pydata/xarray/discussions/6406) when the backend was first being designed.

For broader coverage of analysis, visualization, and distributed execution, see the [xarray documentation](https://docs.xarray.dev) and the [Dask documentation](https://docs.dask.org).
