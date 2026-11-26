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
- `dtype`: data type on disk (for example `np.float32`, `"<f4"`, `">f8"`). The values come back in the machine's native byte order, as with xarray's netCDF and zarr backends, so a `">f8"` file reads as `float64`
- `coords`: mapping of dimension name to coordinate array, in on-disk order
- `name`: variable name
- `attrs`: optional attributes of the Dataset, not of the variable. When several files are opened together, xarray keeps the first file's, so give every file the same ones. Attributes of a variable, such as its `units`, can be set after loading (`dataset["ux"].attrs["units"] = "m/s"`); setting them from the read specs is tracked in [#91](https://github.com/fschuch/xarray-binfile/issues/91)
- `order`: memory layout of the file, `"C"` (default, last dimension varies fastest, as NumPy's `tofile` and C programs write) or `"F"` (first dimension varies fastest, as Fortran programs such as 2DECOMP&FFT and Xcompact3d write). Both describe the same bytes: a Fortran file with dims `(x, y, z)` is also a C file with dims `(z, y, x)`. Pick the one that keeps your dimension names in their natural order
- `coord_attrs`: optional attributes per coordinate, attached on read (for example `{"x": {"units": "m"}}`)

The backend validates the file size against `coords` and `dtype` before reading anything, so a wrong number of elements or item size fails early. It cannot catch a wrong byte order, or a shape with the same number of elements (two dimensions swapped, for example): those open and read garbage.

## Write protocol

The write protocol is `WriteSpecsGetterProtocol`.

Contract:

- input: `xarray.DataArray`
- output: `Iterator[WriteSpecs]`

`WriteSpecs` fields:

- `filename`: output path. A relative path is resolved against the directory passed to `to_file`, which must exist, and may include sub-folders, which are created on demand, but it must stay inside that directory (`..` escaping it is rejected; symbolic links are not followed for this check). An absolute path is used as is, so one call can target several locations. A symbolic link at the destination is followed: the file it points to is replaced and the link stays. New files get the permissions the umask allows
- `sub_array`: array slice to write into that file, already transposed to on-disk order
- `dtype`: optional on-disk data type; when set, the sub-array is cast right before serialization (including byte order, for example `"<f4"`). The cast may narrow within a kind (`f8` to `f4`, `i8` to `i4`, with no range check) or widen (`i4` to `f8`), but `to_file` refuses, before writing any file, a cast that changes the kind: float to integer would truncate and turn NaN into 0, and complex to float would drop the imaginary part. Cast the array yourself first if that is what you want. When omitted, the in-memory dtype and native byte order are written as-is, so make sure they match what your read specs getter declares.
- `order`: memory layout of the file, `"C"` (default) or `"F"`, with the same meaning as on `ReadSpecs`. `tofile` always serializes in C order, so Fortran order is obtained by flattening the array first.

`to_file` also accepts a `progress` wrapper applied to the iterator of write specs, for example `progress=tqdm` or `progress=functools.partial(tqdm, desc="ux")`, to report progress without the library depending on a progress-bar package.

```{warning}
Writes are eager and whole-file only. Each `sub_array` is fully loaded into memory (a Dask compute for lazy data) and its file is written in a single pass — the backend never appends to, patches, or resumes a file. Files are staged in a temporary file next to their destination and atomically moved into place with `os.replace` once complete, so an interrupted write never leaves a truncated file behind. That protects against partially-updated, corrupted files, but it means every `sub_array` you yield must fit in memory. Split large arrays into more, smaller files, and reach for xarray-supported formats such as NetCDF or Zarr when you need streaming or partial writes.
```

See the [API reference](../references/api-reference.rst) for the authoritative signatures.

## Shipped conventions

The module `xarray_binfile.conventions` provides small, ready-to-use implementations of both protocols. Each convention exposes a `reader` (the read specs getter) and a `writer` (the write specs getter) built from the same two ingredients, so files written by one can always be read back by the other:

- `FilenamePattern`: one `str.format`-style template such as `"{name}-{step:04d}.bin"`. The regular expression used for reading is derived from the template, so the two cannot drift apart. Matching is anchored to the whole filename, and a zero-fill width is a minimum width (step `12345` written with `04d` still reads back). Templates without a separator between a name ending in a digit and the number are ambiguous under that rule (`phi1000` reads as `phi` + `1000`); pass `exact_width=True` to match exactly the declared width instead (`phi1` + `000`). Either way, writing refuses any filename that would read back as other values (a name ending in a digit by default, a step wider than the template with `exact_width=True`, `phi1` at `0.25` written as `phi10.250` by `"{name}{time:.3f}"`), so a separator such as `-` between the fields is the safest choice. `glob()` turns the template into a pattern for `pathlib.Path.glob` (`"*-*.bin"`, or `"ux-*.bin"` with `glob(name="ux")`), and a `name` value may start with folder segments (`"geometry/epsi"`), which are kept in front of the formatted filename.
- `Layout`: the dimension order, coordinate values, dtype and memory `order` of one file on disk, plus optional `coord_attrs`. Readers attach these coordinates to every file; writers refuse arrays that do not match them.

Every shipped convention inherits `Convention`, the base class that turns a `reader`/`writer` pair into a full convention: `name_of_file(path)`, `accepts(path)`, `files(directory)`, `stack(dataset)` and `open(directory, ...)`, all derived from `reader` by default and overridden where something cheaper exists. Subclass it for your own conventions; a bare object with only `reader` and `writer` is wrapped through `Convention.adapt` when it joins a `FolderConventions` or `PatternConventions`, so it gets the same defaults.

Every shipped convention also offers:

- `files(directory)`: the files in a folder that follow its pattern, and `open(directory, ...)`: those files as one lazy dataset through `xarray.open_mfdataset`, with `variables=[...]` to open a subset and any other keyword (`chunks`, `parallel`) forwarded. Without `chunks`, files larger than Dask's `array.chunk-size` are split into slabs along their slowest axis on disk, and smaller files stay one chunk each; see {ref}`Plan chunks for how you read the data <plan-chunks>`. Discovery is driven by the anchored pattern only, so unrelated files in the data folder (an XDMF index, notes, backups, hidden files, the temporary files of an interrupted write) are never opened
- `names`: the variable names the convention accepts. When the pattern alone is too permissive, for example a bare `"{name}"` template for extension-less files that would also match `README` or `Makefile`, `names=("epsilon",)` restricts reading, discovery and writing to the listed variables, and lets a `PatternConventions` move on to the next member
- `stacks`: dimensions encoded in the variable names, see [Stacked variables](#stacked-variables)
- `name_of`: a hook returning the name to write an array under, in place of `data_array.name` (for example `lambda da: da.attrs["file_name"]`)
- `time_dtype` (time-series conventions): the dtype of the `time` coordinate on read, for example `np.float32` to match single-precision data. Most operations keep float32 data in float32 with either time dtype; a float32 time matters when the time values enter the arithmetic, as in `integrate("time")` or `u * u.time`, which otherwise give float64 results. With `StepIndexedFiles`, float32 times write back to their steps within a few units in the last place.

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

```python continuation
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

```python continuation
from xarray_binfile.conventions import TimeStampedFiles

convention = TimeStampedFiles(layout, pattern="{name}-{time:.3f}.bin")
```

The `time` coordinate is the float parsed from the filename. On write, each value is formatted with the pattern precision, and arrays whose values collide at that precision are rejected.

### Static fields

Fields without a time dimension, such as a geometry or a mask, use one file per variable:

```python continuation
from xarray_binfile.conventions import StaticFiles

static = StaticFiles(layout, pattern="{name}.bin")
```

### Stacked variables

Solvers often store a vector field as one file per component (`ux`, `uy`, `uz`) and a set of scalar fractions as numbered variables (`phi1`, `phi2`). A `VariableStack` declares that encoding once: the dimension it stands for, the template that builds a variable name from `{name}` and the value, and the values it may take. Conventions split arrays along those dimensions on write and stack the variables back on read:

```python continuation fixture:dataset fixture:output_dir
from xarray_binfile.conventions import VariableStack

velocity = VariableStack("i", "{name}{i}", values=("x", "y", "z"))
scalars = VariableStack("n", "{name}{n:d}", values=range(1, 10), names=("phi",))
convention = StepIndexedFiles(layout, stacks=[velocity, scalars])

# u with i = ["x", "y", "z"] is written as ux-0001.bin, uy-0001.bin, uz-0001.bin
dataset.binary_engine.to_file(convention.writer, output_dir)

# ux, uy, uz found on disk come back as u with the coordinate i
opened = convention.open(output_dir)          # stacks by default
raw = convention.open(output_dir, stack=False)  # ux, uy, uz as on disk
```

`values` is required because templates without a separator match almost any name (`pp` fits `"{name}{i}"` as `p` + `p`); it also fixes the order of the stacked coordinate, and writing a coordinate value that is not listed raises. On read each value is matched literally, so `vortx` is `vort` + `x`, and `names` (any collection, kept as a `frozenset`; a bare string is rejected) says which base names to reassemble; without it, a group needs `min_components` (two by default) components, so a lone `vorticity` is not read as `vorticit` + `y`. Stacking refuses to overwrite a variable that already carries the base name. Writing always splits an array carrying the dimension, whatever its name. The same declarations work on their own through `VariableStack.stack`, `stack_variables` and `split_variables`. See the [stacked variables tutorial](../tutorials/stacked-variables.ipynb).

### Mixed conventions in one folder

When files following different conventions share a folder, for example `ux-0001.bin` next to `epsi.bin`, `PatternConventions` tries its members in order on read and keeps the first whose pattern accepts the filename, so list the most specific pattern first. On write it offers the array to every member and requires exactly one to accept it, based on dimensions and coordinates:

```python continuation fixture:dataset fixture:output_dir
from xarray_binfile.conventions import PatternConventions

conventions = PatternConventions([StepIndexedFiles(layout), StaticFiles(layout)])
```

### Shapes that depend on the folder

Projects often keep arrays of different shapes in different folders:

```text
caseC/
	xy_planes/
		uxy-0001.bin
	3d/
		ux-0001.bin
	static/
		epsi.bin
```

`FolderConventions` dispatches to one convention per folder. Reading picks the convention whose folder matches the end of the file's parent path, preferring the most specific folder when several match (`snapshots/3d` over `3d`). The dataset root is registered as `"."` and matches any file not claimed by a more specific folder. Writing offers the array to every convention and requires exactly one to accept it, based on the array dimensions and coordinates, then writes into that folder. If no convention or more than one accepts the array, a `LayoutMismatchError` is raised: the backend never guesses. An array whose name starts with a registered folder (`"static/epsi"`) is sent to that folder's convention directly, under the remaining name, which resolves the ambiguity between folders sharing one layout. `files(directory)` and `open(directory)` walk every registered folder; `PatternConventions` can be a member, for example as the root convention.

Reading keeps only the filename's name, not its folder, so `xy_planes/ux-0001.bin` and `3d/ux-0001.bin` would both be `ux`. One name may span several folders when its files have the same dimensions at different positions, for example one run split in time across `early/` and `late/`, or tiles of one grid. Otherwise `open()` refuses the files before opening them: a name whose files have different dimensions (a plane and a volume) cannot be one variable, and xarray places files by the first value of every coordinate, so two files at the same position (the same step in two folders, or grids that start at the same point) would silently keep only one of them. Dates are compared in nanoseconds, as xarray does. Give such files distinct names, as in the tree above, or open each folder with its own convention. The check runs in `open()`; a direct `xr.open_mfdataset(..., read_specs_getter=conventions.reader)` call opens one file at a time and cannot detect it. Keeping the folder in the variable name is tracked in [#87](https://github.com/fschuch/xarray-binfile/issues/87).

```python continuation fixture:dataset fixture:output_dir fixture:case_dir
from xarray_binfile.conventions import FolderConventions

x, y, z = np.arange(64), np.arange(32), np.arange(16)
conventions = FolderConventions(
    {
        "xy_planes": StepIndexedFiles(Layout({"x": x, "y": y}, dtype="<f4")),
        "3d": StepIndexedFiles(Layout({"x": x, "y": y, "z": z}, dtype="<f4")),
        "static": StaticFiles(Layout({"x": x, "y": y, "z": z}, dtype="<f4")),
    }
)

dataset = conventions.open(case_dir)  # or xr.open_mfdataset(conventions.files(case_dir), engine="binfile", read_specs_getter=conventions.reader)
dataset.binary_engine.to_file(conventions.writer, output_dir)
```

### Validation on write

Every shipped writer checks the array against its `Layout` before yielding anything:

- dimension names must match the layout plus the split dimension (`time` by default)
- sizes along each dimension must match
- coordinate values must be equal to the layout coordinates, unless `check_coords=False`

It also requires the array to be named, refuses names or values that would produce a filename the paired reader cannot parse back (for example a variable called `u.x` with the default pattern), and refuses to proceed when two slices would map to the same filename. Mismatches raise `LayoutMismatchError` (a `ValueError`) with the offending dimension in the message.

`to_file` collects every write spec of the call, for all variables of a Dataset, before writing the first file. It refuses the call when two specs map to the same file, for example a stacked `u` whose x component is `ux-0000.bin` next to a separate variable `ux`, or `U` and `u` where the filesystem ignores case (the default on macOS and Windows). Paths are compared with symbolic links followed, in folders and at the file itself, so a link and the file it points to are one file, and names that differ only in Unicode form (`café` with a composed `é` or with `e` and a combining accent) also count as one file where the filesystem ignores that, as macOS does. Any other error from a writer also stops the call before anything is written.

Each `to_file` call is independent: writing a file that already exists replaces it. A split dimension without a coordinate is numbered by position from 0 in every call, so a loop that writes one `snapshot.expand_dims("time")` per step writes every snapshot to the same file, and only the last one is kept. Give the dimension its value instead, with `snapshot.expand_dims(time=[t])`. With `time_step` set, positions are also read as time values (position 1 is step 10 with `time_step=0.1`), another reason to set the coordinate.

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
- Keep dimension order explicit and consistent, and declare the memory `order` (`"C"` or `"F"`) that matches the program that wrote the files.
- Make dtype and endianness explicit.
- Fail loudly on layout mismatch instead of broadcasting or squeezing.
- Test read/write round-trips (`xarray.testing.assert_identical`).

```{warning}
Raw binaries are machine-specific unless conventions are explicit. Endianness (little-endian vs big-endian), word size, and layout assumptions can silently corrupt interpretation. Validate dtype and byte order when moving files across systems.
```

## Related pages

- [Built-in tutorial: reading](../tutorials/read.ipynb)
- [Built-in tutorial: writing](../tutorials/write.ipynb)
- [Built-in tutorial: stacked variables](../tutorials/stacked-variables.ipynb)
- [API reference](../references/api-reference.rst)
