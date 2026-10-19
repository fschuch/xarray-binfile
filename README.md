# Xarray-binfile

<p align="center">
<a href="https://github.com/fschuch/xarray-binfile"><img src="https://raw.githubusercontent.com/fschuch/xarray-binfile/refs/heads/main/docs/logo.png" alt="Xarray-binfile logo" width="320"></a>
</p>
<p align="center">
    <em>Custom Xarray file engine to handle raw binary files</em>
</p>

______________________________________________________________________

- QA:
  [![CI](https://github.com/fschuch/xarray-binfile/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/fschuch/xarray-binfile/actions/workflows/ci.yaml)
  [![CodeQL](https://github.com/fschuch/xarray-binfile/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/fschuch/xarray-binfile/actions/workflows/github-code-scanning/codeql)
  [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/fschuch/xarray-binfile/main.svg)](https://results.pre-commit.ci/latest/github/fschuch/xarray-binfile/main)
  [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=fschuch_xarray-binfile&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=fschuch_xarray-binfile)
  [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=fschuch_xarray-binfile&metric=coverage)](https://sonarcloud.io/summary/new_code?id=fschuch_xarray-binfile)
  [![CodeFactor](https://www.codefactor.io/repository/github/fschuch/xarray-binfile/badge)](https://www.codefactor.io/repository/github/fschuch/xarray-binfile)

- Docs:
  [![Docs](https://github.com/fschuch/xarray-binfile/actions/workflows/docs.yaml/badge.svg?branch=main)](https://docs.fschuch.com/xarray-binfile)

- Package:
  [![PyPI - Version](https://img.shields.io/pypi/v/xarray-binfile.svg?logo=pypi&label=PyPI)](https://pypi.org/project/xarray-binfile/)
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xarray-binfile.svg?logo=python&label=Python)](https://pypi.org/project/xarray-binfile/)

- Meta:
  [![Wizard Template](https://img.shields.io/badge/Wizard-Template-%23447CAA)](https://github.com/fschuch/wizard-template)
  [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
  [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  ![GitHub License](https://img.shields.io/github/license/fschuch/xarray-binfile?color=blue)
  [![EffVer Versioning](https://img.shields.io/badge/version_scheme-EffVer-0097a7)](https://jacobtomlinson.dev/effver)

______________________________________________________________________

## Overview

Xarray-binfile is a Python package that integrates raw binary files with xarray. You provide the metadata that describes each binary file, and the package exposes those files through xarray datasets and data arrays with optional lazy chunking through Dask.

The package was first designed around binary outputs from the Fortran framework [2DECOMP&FFT](https://github.com/2decomp-fft/2decomp-fft) and the CFD solver [Xcompact3d](https://github.com/xcompact3d/Incompact3d), which is built on top of it. It is not limited to those ecosystems and can be used with any raw binary convention once the corresponding metadata mapping is provided.

It is aimed at workflows where raw binary files are part of an existing convention, such as simulation outputs or NumPy `.tofile()` dumps. Once opened, the data can be indexed, reduced, plotted, and computed with the usual xarray and Dask APIs. See [why not just `numpy.fromfile`?](https://docs.fschuch.com/xarray-binfile/getting-started/about.html#why-not-just-numpy-fromfile) in the documentation for what that buys you over `numpy.fromfile` and `numpy.memmap`.

> [!CAUTION]
> Raw binary files are machine- and convention-dependent. Differences such as little-endian vs big-endian byte order, word size, record layout, and memory ordering can change how bytes should be interpreted. Always verify dtype and endianness when sharing files across machines or toolchains.

## Installation

```bash
pip install xarray-binfile
```

Python 3.10 or later is required; xarray, Dask, and NumPy are installed as dependencies. See the [installation guide](https://docs.fschuch.com/xarray-binfile/getting-started/installation.html) for version pinning and for `pyproject.toml`, `requirements.txt`, and conda environment examples.

## Quick start

Describe the file convention once, then let xarray and Dask do the rest:

```python
import numpy as np

from xarray_binfile.conventions import Layout, StepIndexedFiles

# Describe the convention once: coordinates, dtype, and filename pattern
convention = StepIndexedFiles(
    layout=Layout(
        {"x": np.linspace(0, 1, 8), "y": np.linspace(0, 1, 4)},
        dtype=np.float32,
    ),
    pattern="{name}-{step:04d}.bin",  # ux-0000.bin, ux-0001.bin, ...
)

# Open every matching file in a folder as one lazy, Dask-backed dataset
ds = convention.open("path/to/data", chunks={"time": 1})

# From here on it is plain xarray
mean = ds["ux"].mean(dim="time").compute()

# Write a derived result back to the same raw layout
mean.rename("ux_mean").expand_dims(time=[0]).binary_engine.to_file(
    convention.writer, "path/to/data"
)
```

The same files also open through the standard xarray entry points, `xr.open_dataset(path, engine="binfile", read_specs_getter=convention.reader)` and `xr.open_mfdataset(...)`, so they fit into existing xarray code.

The documentation covers the rest:

- [Spec getter protocols](https://docs.fschuch.com/xarray-binfile/usage/spec-getter-protocols.html): the shipped conventions (step-indexed, time-stamped, static files, folder layouts, stacked variables) and how to describe your own.
- [Read lazily with Dask](https://docs.fschuch.com/xarray-binfile/tutorials/read.html) and [write derived results](https://docs.fschuch.com/xarray-binfile/tutorials/write.html).
- [Parallel and larger-than-memory](https://docs.fschuch.com/xarray-binfile/tutorials/parallel-and-larger-than-memory.html) workflows, [benchmarking](https://docs.fschuch.com/xarray-binfile/tutorials/benchmarking.html) on your own data, and [practical examples](https://docs.fschuch.com/xarray-binfile/tutorials/examples.html).
- [API reference](https://docs.fschuch.com/xarray-binfile/references/api-reference.html).

## Contributing

Bug reports, feature requests, and pull requests are welcome. Start with [CONTRIBUTING.md](https://github.com/fschuch/xarray-binfile/blob/main/CONTRIBUTING.md) for the development setup with Hatch and the pull request checklist, or read the [full contributor guide](https://docs.fschuch.com/xarray-binfile/references/how-to-contribute.html) in the documentation. This project follows the [Contributor Covenant](https://github.com/fschuch/xarray-binfile/blob/main/CODE_OF_CONDUCT.md).

## Copyright and License

© 2025 [Felipe N. Schuch](https://github.com/fschuch).
All content is under [MIT License](https://github.com/fschuch/xarray-binfile/blob/main/LICENSE).
