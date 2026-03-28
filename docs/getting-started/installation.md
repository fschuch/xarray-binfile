# Installation

## Install from PyPI

`xarray-binfile` depends on both xarray and Dask at runtime, because its primary workflow is lazy, chunked access to raw binary datasets.

```bash
pip install xarray-binfile
```

## Pin a version

```bash
pip install "xarray-binfile==0.1.0b0"
```

## Install from GitHub

```bash
pip install git+https://github.com/fschuch/xarray-binfile.git
```

## Add it to environment files

::::{tab-set}

:::{tab-item} `requirements.txt`

```text
xarray-binfile
```

:::

:::{tab-item} `pyproject.toml`

```toml
[project]
dependencies = [
  "xarray-binfile",
]
```

:::

:::{tab-item} `environment.yml`

```yaml
name: xarray-binfile-example
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - xarray-binfile
```

:::

::::

## Development installation

The project uses Hatch for development, testing, and docs builds.

```bash
git clone https://github.com/fschuch/xarray-binfile.git
cd xarray-binfile
hatch env create
hatch run qa
```

```{tip}
If you want Hatch environments stored inside the repository, run `hatch config set dirs.env.virtual .venv` before creating the environments.
```

## What else do you need?

At runtime, the package expects you to provide metadata for your binary layout. In practice that means you will usually define:

- the dtype stored in each file
- the coordinates and dimension order for each array
- a filename convention that can be parsed back into metadata such as variable name and timestep

The [about page](about.md) explains that model, and the usage tutorials show complete examples built on the bundled tutorial helpers.
