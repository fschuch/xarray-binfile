# Installation

As a Python package, `xarray-binfile` can be installed with pip from PyPI or directly from GitHub. The package depends on both xarray and Dask at runtime, because its primary workflow is lazy, chunked access to raw binary datasets.

You can use your preferred environment management tool, activate an existing environment, and then install the package with pip:

```bash
pip install xarray-binfile
```

The command above installs the latest release from PyPI. To reproduce an environment later, pin the exact version you need by adding a version specifier, for example `pip install "xarray-binfile==X.Y.Z"`.

You can also use any dependency management tool that supports pip-style dependencies, such as Poetry, Hatch, uv, or Conda:

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
