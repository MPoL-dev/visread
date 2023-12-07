# Visread documentation

[![Tests](https://github.com/MPoL-dev/visread/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/MPoL-dev/visread/actions/workflows/tests.yml)
[![gh-pages docs](https://github.com/MPoL-dev/visread/actions/workflows/gh_docs.yml/badge.svg)](https://mpol-dev.github.io/visread/)


**What is this package**? This package is really three things:
1) a collection of documentation and tutorials demonstrating how to read visibilities (and associated metadata) from a calibrated CASA Measurement Set file into memory using CASA tools like `table` and `ms`. For this, you don't even need to concern yourself with installing visread package, just browse the tutorials directly.
2) a few routines within the visread package that codify the data access patterns shown in the tutorials, available via `pip install visread[casa]` and described further in \[installation.md\]. These routines depend on the `casatools` package, and thus have the same Python version and operating system requirements that \[Modular CASA\](<https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Compatibility>) does. Depending on the state of the release cycle, these can be quite constraining and preclude using modern Python versions.
3) a set of non-imaging analysis and visualization routines to work with the visibility data, available via `pip install visread` and described further in \[installation.md\]. These core routines *do not* have a `casatools` dependency, and so they should be usuable in all current Python versions.

**To get started**, you can get some ideas for working with `table` and `ms` directly from the *Introduction to CASA tools*. For convenience, you can install the *visread* package to provide some lightweight routines for common visibility manipulations, which are built on top of the core `casatools` functionality.

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

tutorials/introduction_to_casatools
tutorials/rescale_AS209_weights
installation.rst
casatools-api.rst
api.rst
```

# Citation

If you find these tutorials or code useful, please cite:

```
@software{visread,
author       = {Ian Czekala and
                Loomis, Ryan and
                Andrews, Sean and
                Huang, Jane and
                Rosenfeld, Katherine},
title        = {MPoL-dev/visread},
month        = jan,
year         = 2021,
publisher    = {Zenodo},
version      = {v0.0.1},
doi          = {10.5281/zenodo.4432501},
url          = {https://doi.org/10.5281/zenodo.4432501}
}
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
