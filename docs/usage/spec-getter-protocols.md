# Spec Getter Protocols for Read and Write

This page explains the protocol interfaces used by xarray-binfile and shows:

- the protocol contracts for read and write operations
- the implementations shipped with the package
- how to implement your own getters for custom binary layouts

## Why spec getters exist

Raw binary files are not self-describing. They do not store enough metadata for xarray to know:

- dimension names and order
- coordinate values
- variable name
- dtype and byte order

xarray-binfile solves this by asking you for callables that provide metadata for reads and file splitting rules for writes.

## File-layout examples

The protocol approach is useful because real projects organize files differently.

### Time-series variables only

```text
caseA/
	ux001.bin
	ux002.bin
	ux003.bin
	uy001.bin
	uy002.bin
	uy003.bin
```

In this layout, the read specs getter usually parses:

- variable name (`ux`, `uy`)
- step index (`001`, `002`, `003`)

and returns `coords` including a single-value `time` coordinate for each file.

### Mixed time-series plus static fields

```text
caseB/
	ux001.bin
	ux002.bin
	ux003.bin
	uy001.bin
	uy002.bin
	uy003.bin
	epsi.bin
```

Here, `ux*` and `uy*` are time-dependent while `epsi.bin` is static.

A custom read specs getter can support this by using two filename rules:

- time-series rule: parse variable + step, include `time=[step]`
- static rule: parse static variable (for example `epsi`), return coords without `time`

In xarray workflows, static variables without a `time` dimension can be naturally broadcast against time-dependent variables during computations.

## Read protocol

The read protocol is `ReadSpecsGetterProtocol`.

Contract:

- input: `pathlib.Path`
- output: `ReadSpecs`

`ReadSpecs` fields:

- `filepath`: path to the binary file
- `dtype`: data type (for example `np.float32`, `"<f4"`, `">f8"`)
- `coords`: mapping of dimension name to coordinate array
- `name`: variable name
- `attrs`: optional attributes

See the API reference for authoritative signatures and behavior:

- `ReadSpecs`: [API reference](../references/api-reference.rst)
- `ReadSpecsGetterProtocol`: [API reference](../references/api-reference.rst)

## Write protocol

The write protocol is `WriteSpecsGetterProtocol`.

Contract:

- input: `xarray.DataArray`
- output: `Iterator[WriteSpecs]`

`WriteSpecs` fields:

- `filename`: output file name
- `sub_array`: array slice to write into that file

See the API reference for authoritative signatures and behavior:

- `WriteSpecs`: [API reference](../references/api-reference.rst)
- `WriteSpecsGetterProtocol`: [API reference](../references/api-reference.rst)

## Implementations shipped with xarray-binfile

The package includes a built-in reference implementation in `xarray_binfile.tutorial.FileSpecsGetter`.

It provides both protocol implementations:

- `FileSpecsGetter.reader`: implements `ReadSpecsGetterProtocol`
- `FileSpecsGetter.writer`: implements `WriteSpecsGetterProtocol`

It uses a common pattern:

- filename convention like `{name}-{digits:04}.bin`
- regex parsing for variable name and time index
- base coordinates plus a `time` coordinate extracted from filename

This shipped helper targets a single time-series naming convention. Mixed layouts (for example `ux001.bin` together with static `epsi.bin`) are typically handled by creating your own getter based on the same protocol contracts.

A second shipped helper, `xarray_binfile.tutorial.DatasetGenerator`, consumes a read specs getter and generates synthetic datasets for tests/tutorials.

These implementations are documented in the API section and should be considered the source of truth:

- `FileSpecsGetter`: [API reference](../references/api-reference.rst)
- `DatasetGenerator`: [API reference](../references/api-reference.rst)

## How to create your own read specs getter

Create a callable that follows `ReadSpecsGetterProtocol`. In practice:

1. Parse your filename convention.
2. Resolve variable identity (for example variable name and step index).
3. Build explicit `coords` and `dtype`.
4. Return a `ReadSpecs` object.

Use `FileSpecsGetter.reader` as the reference behavior and adapt only what differs for your project.

## How to create your own write specs getter

Create a callable that follows `WriteSpecsGetterProtocol`. In practice:

1. Decide how one in-memory array maps to one or more output files.
2. Yield one `WriteSpecs` object per output file.
3. Ensure each `sub_array` is transposed to the expected on-disk dimension order.
4. Use deterministic filename rules that your read getter can parse back.

Use `FileSpecsGetter.writer` as the reference behavior and adapt only what differs for your project.

## Validation checklist for custom implementations

- Validate filename parsing with explicit errors.
- Keep dimension order explicit and consistent.
- Make dtype and endianness explicit.
- Test read/write round-trips (`xarray.testing.assert_allclose`).
- Start from `FileSpecsGetter` and replace only what differs.

```{warning}
Raw binaries are machine-specific unless conventions are explicit. Endianness (little-endian vs big-endian), word size, and layout assumptions can silently corrupt interpretation. Validate dtype and byte order when moving files across systems.
```

## Related pages

- [Built-in tutorial: reading](../tutorials/read.ipynb)
- [Built-in tutorial: writing](../tutorials/write.ipynb)
- [API reference](../references/api-reference.rst)
