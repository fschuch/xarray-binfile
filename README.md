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

It is aimed at workflows where raw binary files are part of an existing convention, such as simulation outputs or NumPy `.tofile()` dumps. Once opened, the data can be indexed, reduced, plotted, and computed with the usual xarray and Dask APIs.

See the documentation for end-to-end examples covering lazy loading, Dask-backed computations, task-graph visualization, writing derived results back to `.bin` files, parallel and larger-than-memory workflows, benchmarking on your own data, batch dtype conversion, probe time-series export to CSV, and quick matplotlib plots.

> [!CAUTION]
> Raw binary files are machine- and convention-dependent. Differences such as little-endian vs big-endian byte order, word size, record layout, and memory ordering can change how bytes should be interpreted. Always verify dtype and endianness when sharing files across machines or toolchains.

## Copyright and License

© 2025–2026 [Felipe N. Schuch](https://github.com/fschuch).
All content is under [MIT License](https://github.com/fschuch/xarray-binfile/blob/main/LICENSE).
