# Spec Getter Protocols for Read and Write

This page explains how xarray-binfile learns about your files. It covers:

- the protocol contracts for read and write operations
- the conventions shipped with the package, and how to compose them
- how to implement your own getters when no shipped convention fits

## Why spec getters exist

Raw binary files are not self-describing. They do not store enough metadata for xarray to know:

- dimension names and order
- coordinate values
- variable name
- dtype and byte order

xarray-binfile solves this by asking you for callables that provide metadata for reads and file-splitting rules for writes. Most projects follow a filename convention such as `<variable><separator><number>.<suffix>`, sometimes with the parent folder indicating the shape of the files. The shipped conventions cover those cases; the protocols let you cover everything else.

## Read protocol

The read protocol is `ReadSpecsGetterProtocol`.

Contract:

- input: `pathlib.Path`
- output: `ReadSpecs`

`ReadSpecs` fields:

- `filepath`: path to the binary file
- `dtype`: data type (for example `np.float32`, `"<f4"`, `">f8"`)
- `coords`: mapping of dimension name to coordinate array, in on-disk order
- `name`: variable name
- `attrs`: optional attributes

The backend validates the file size against `coords` and `dtype` before reading anything, so a wrong shape or byte order fails early instead of producing garbage.

## Write protocol

The write protocol is `WriteSpecsGetterProtocol`.

Contract:

- input: `xarray.DataArray`
- output: `Iterator[WriteSpecs]`

`WriteSpecs` fields:

- `filename`: output path. A relative path is resolved against the directory passed to `to_file`, which must exist, and may include sub-folders, which are created on demand, but it must stay inside that directory (`..` escaping it is rejected; symbolic links are not followed for this check). An absolute path is used as is, so one call can target several locations
- `sub_array`: array slice to write into that file, already transposed to on-disk order
- `dtype`: optional on-disk data type; when set, the sub-array is cast right before serialization (including byte order, for example `"<f4"`). When omitted, the in-memory dtype and native byte order are written as-is, so make sure they match what your read specs getter declares.

```{warning}
Writes are eager and whole-file only. Each `sub_array` is fully loaded into memory (a Dask compute for lazy data) and its file is written in a single pass — the backend never appends to, patches, or resumes a file. Files are staged in a temporary file next to their destination and atomically moved into place with `os.replace` once complete, so an interrupted write never leaves a truncated file behind. That protects against partially-updated, corrupted files, but it means every `sub_array` you yield must fit in memory. Split large arrays into more, smaller files, and reach for xarray-supported formats such as NetCDF or Zarr when you need streaming or partial writes.
```

See the [API reference](../references/api-reference.rst) for the authoritative signatures.

## Shipped conventions

The module `xarray_binfile.conventions` provides small, ready-to-use implementations of both protocols. Each convention exposes a `reader` (the read specs getter) and a `writer` (the write specs getter) built from the same two ingredients, so files written by one can always be read back by the other:

- `FilenamePattern`: one `str.format`-style template such as `"{name}-{step:04d}.bin"`. The regular expression used for reading is derived from the template, so the two cannot drift apart. Matching is anchored to the whole filename, and a zero-fill width is a minimum width (step `12345` written with `04d` still reads back).
- `Layout`: the dimension order, coordinate values and dtype of one file on disk. Readers attach these coordinates to every file; writers refuse arrays that do not match them.

### Time-series numbered by step

```text
caseA/
	ux-0001.bin
	ux-0002.bin
	uy-0001.bin
	uy-0002.bin
```

```python
import numpy as np
from xarray_binfile.conventions import Layout, StepIndexedFiles

layout = Layout({"x": np.arange(64), "y": np.arange(32), "z": np.arange(16)}, dtype="<f4")
convention = StepIndexedFiles(layout, pattern="{name}-{step:04d}.bin")
```

By default the `time` coordinate is the integer step. When the interval between snapshots is known, pass `time_step` and the coordinate becomes `step * time_step` on read, while on write the coordinate is divided back into a step. Values that do not land on an integer step raise instead of being truncated, so two snapshots can never silently overwrite each other:

```python
convention = StepIndexedFiles(layout, pattern="{name}.{step:06d}", time_step=0.05)
```

Other spellings of the same idea only change the pattern: `"{name}{step:03d}.dat"` for `ux001.dat`, or `"{name}_{step:d}.bin"` for an unpadded counter.

### Time-series stamped with the physical time

Some solvers write the time itself in the filename:

```text
caseB/
	ux-0.250.bin
	ux-0.500.bin
```

```python
from xarray_binfile.conventions import TimeStampedFiles

convention = TimeStampedFiles(layout, pattern="{name}-{time:.3f}.bin")
```

The `time` coordinate is the float parsed from the filename. On write, each value is formatted with the pattern precision, and arrays whose values collide at that precision are rejected.

### Static fields

Fields without a time dimension, such as a geometry or a mask, use one file per variable:

```python
from xarray_binfile.conventions import StaticFiles

static = StaticFiles(layout, pattern="{name}.bin")
```

### Shapes that depend on the folder

Projects often keep arrays of different shapes in different folders:

```text
caseC/
	xy_planes/
		ux-0001.bin
	3d/
		ux-0001.bin
	static/
		epsi.bin
```

`FolderConventions` dispatches to one convention per folder. Reading picks the convention whose folder matches the end of the file's parent path, preferring the most specific folder when several match (`snapshots/3d` over `3d`). Writing offers the array to every convention and requires exactly one to accept it, based on the array dimensions and coordinates, then writes into that folder. If no convention or more than one accepts the array, a `LayoutMismatchError` is raised: the backend never guesses.

```python
from xarray_binfile.conventions import FolderConventions

x, y, z = np.arange(64), np.arange(32), np.arange(16)
conventions = FolderConventions(
    {
        "xy_planes": StepIndexedFiles(Layout({"x": x, "y": y}, dtype="<f4")),
        "3d": StepIndexedFiles(Layout({"x": x, "y": y, "z": z}, dtype="<f4")),
        "static": StaticFiles(Layout({"x": x, "y": y, "z": z}, dtype="<f4")),
    }
)

dataset = xr.open_mfdataset(
    sorted(case_dir.rglob("*.bin")), engine="binfile", read_specs_getter=conventions.reader
)
dataset.binary_engine.to_file(conventions.writer, output_dir)
```

### Validation on write

Every shipped writer checks the array against its `Layout` before yielding anything:

- dimension names must match the layout plus the split dimension (`time` by default)
- sizes along each dimension must match
- coordinate values must be equal to the layout coordinates, unless `check_coords=False`

It also requires the array to be named, refuses names or values that would produce a filename the paired reader cannot parse back (for example a variable called `u.x` with the default pattern), and refuses to proceed when two slices would map to the same filename. Mismatches raise `LayoutMismatchError` (a `ValueError`) with the offending dimension in the message.

```{note}
`xarray_binfile.tutorial.FileSpecsGetter` is the previous reference implementation. It is deprecated in favor of `StepIndexedFiles` and kept unchanged for backward compatibility. Replace `FileSpecsGetter(base_coords=coords, dtype=dtype)` with `StepIndexedFiles(Layout(coords, dtype=dtype))`; the default pattern `"{name}-{step:04d}.bin"` produces the same filenames.
```

## How to create your own convention

Start from the shipped convention closest to your files and change only what differs. When none fits, implement the two protocols directly; `FilenamePattern` and `Layout` are reusable on their own.

Read specs getter:

1. Parse your filename convention (and the parent folder, if it carries information).
1. Resolve variable identity (for example variable name and step index).
1. Build explicit `coords` and `dtype`.
1. Return a `ReadSpecs` object.

Write specs getter:

1. Validate the array against the on-disk layout and raise on mismatch.
1. Decide how one in-memory array maps to one or more output files.
1. Yield one `WriteSpecs` object per output file, with `sub_array` transposed to the on-disk dimension order.
1. Keep each `sub_array` small enough to fit in memory, since it is fully materialized before its file is written.
1. Use deterministic filename rules that your read getter can parse back.

## Validation checklist for custom implementations

- Derive the read and write filename rules from one definition.
- Anchor filename matching to the whole name so stale or backup files are rejected.
- Keep dimension order explicit and consistent.
- Make dtype and endianness explicit.
- Fail loudly on layout mismatch instead of broadcasting or squeezing.
- Test read/write round-trips (`xarray.testing.assert_identical`).

```{warning}
Raw binaries are machine-specific unless conventions are explicit. Endianness (little-endian vs big-endian), word size, and layout assumptions can silently corrupt interpretation. Validate dtype and byte order when moving files across systems.
```

## Related pages

- [Built-in tutorial: reading](../tutorials/read.ipynb)
- [Built-in tutorial: writing](../tutorials/write.ipynb)
- [API reference](../references/api-reference.rst)
