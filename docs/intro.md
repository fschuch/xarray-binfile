# Xarray-binfile

```{centered} Read and write raw binary files using the familiar interface from the Xarray library.
```

Xarray-binfile provides an xarray backend for raw binary files that are not self-describing on their own. You supply the metadata needed to interpret each file, and xarray-binfile handles lazy loading, chunked reads with Dask, and writing results back to raw binary when you need to interoperate with an existing file layout.

The main user story is straightforward:

- describe your file naming and coordinates with a callable compatible with `ReadSpecsGetterProtocol`
- open one file with `xr.open_dataset(..., engine="binfile")` or a collection of files with `xr.open_mfdataset(..., engine="binfile")`
- keep computations lazy with Dask until you call `.compute()` or `.load()`
- optionally write derived arrays back to `.bin` files with `.binary_engine.to_file(...)`

::::{grid} 1 1 2 3
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: getting-started/about
:link-type: doc
:class-header: bg-light

About The Backend
^^^

Learn how xarray-binfile maps raw binary files into labeled xarray datasets and data arrays.

:::

:::{grid-item-card}
:link: getting-started/installation
:link-type: doc
:class-header: bg-light

Install And Configure
^^^

Install the package, understand the runtime dependencies, and get ready to open binary datasets lazily.

:::

:::{grid-item-card}
:link: tutorials/read
:link-type: doc
:class-header: bg-light

Read Lazily With Dask
^^^

Open binary files with chunking, inspect the Dask graph, and compute only the results you need.

:::

:::{grid-item-card}
:link: tutorials/write
:link-type: doc
:class-header: bg-light

Write Derived Results
^^^

Take a computed xarray result and write it back to raw binary files when you need to match an external format.

:::

:::{grid-item-card}
:link: references/api-reference
:link-type: doc
:class-header: bg-light

API Reference
^^^

See the public modules, metadata protocols, backend entrypoint, and tutorial helpers.

:::

:::{grid-item-card}
:link: references/further-reading
:link-type: doc
:class-header: bg-light

Further Reading
^^^

Go deeper on xarray, Dask, NumPy binary I/O, and related simulation workflows.

:::

::::

```{important}
Raw binary files usually need external metadata such as shape, coordinate values, dtype, and filename conventions. Xarray-binfile does not try to infer that information magically. Instead, it asks you to provide it explicitly so the backend can remain predictable and interoperable.
```

```{note}
For portable archival or data exchange workflows, prefer standard xarray-supported formats such as NetCDF or Zarr. Raw binary output is most useful when you need to read or write an existing binary convention used by a simulation code or downstream tool.
```
