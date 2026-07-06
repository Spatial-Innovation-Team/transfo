# transfo

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/Spatial-Innovation-Team/transfo/test.yaml?branch=main

[badge-docs]: https://app.readthedocs.org/projects/transfo/badge/

Ergonomic in-memory transformations operations , using NGFF for read/write

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install transfo:

<!--
1) Install the latest release of `transfo` from [PyPI][]:

```bash
pip install transfo
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/Spatial-Innovation-Team/transfo.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Developer Guide

### Development Environment

This project uses [Hatch](https://hatch.pypa.io/latest/) for managing development environment, for building and for
testing. The development environment is configured to use
`.venv` as the virtual environment directory and includes both `dev` and `test` dependency groups.

#### Setting up the Development Environment

To set up your development environment, run the following at repo root:

```bash
hatch run create-init-dev-env
```

This command will:

1. If no environment exists: Create a virtual environment in `.venv`, install the package in development mode, install
   all development and test dependencies
2. Install or update pre-commit hooks

#### Available Hatch Commands

Once your environment is set up, you can use the following commands:

**Testing:**

```bash
# Run tests
hatch run dev:test

# Run tests with coverage
hatch run dev:test-cov
```

**Linting and Formatting:**

```bash
# Run pre-commit on staged files
hatch run dev:lint-format-staged

# Run pre-commit on all files
hatch run dev:lint-format-all-files

# Install pre-commit hooks manually (if needed)
hatch run dev:pre-commit-install
```

**Entering the Environment:**

```bash
# Activate the virtual environment
hatch shell dev
```

#### Virtual Environment Location

The development environment is created in the `.venv` directory at the project root. This location is specified in the
`[tool.hatch.envs.dev]` configuration in `pyproject.toml`.

#### Dependency Groups

The development environment includes two dependency groups:

- `dev`: Development tools including pre-commit, prek, twine, and ty
- `test`: Testing tools including pytest, coverage, and pytest-cov

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv

[hatch]: https://hatch.pypa.io/

[scverse discourse]: https://discourse.scverse.org/

[issue tracker]: https://github.com/Spatial-Innovation-Team/transfo/issues

[tests]: https://github.com/Spatial-Innovation-Team/transfo/actions/workflows/test.yaml

[documentation]: https://transfo.readthedocs.io

[changelog]: https://transfo.readthedocs.io/page/changelog.html

[api documentation]: https://transfo.readthedocs.io/page/api.html

[pypi]: https://pypi.org/project/transfo
