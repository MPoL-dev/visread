# Visread documentation

[![Tests](https://github.com/MPoL-dev/visread/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/MPoL-dev/visread/actions/workflows/test.yml)
[![gh-pages docs](https://github.com/MPoL-dev/visread/actions/workflows/gh_docs.yml/badge.svg)](https://mpol-dev.github.io/visread/)

This package is really three things:
1) a collection of documentation and tutorials demonstrating how to read visibilities (and associated metadata) from a calibrated CASA Measurement Set file into memory using CASA tools like `table` and `ms`. For this, you don't need to install the visread package, just start browsing the tutorials directly: {ref}`intro-casatools-label`, {ref}`AS209-label`, and {ref}`casatools-api-label`.
2) a few routines within the visread package that codify the data access patterns sketched out in the `casatools` tutorials. These routines depend on the `casatools` package, and thus have the same Python version and operating system requirements that [Modular CASA](<https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Compatibility>) does. Depending on the CASA release cycle, these constraints can preclude modern Python versions. Therefore, these routines are designed to be an *optional* feature of visread and are only installed with `pip install 'visread[casa]'`.
3) a set of non-imaging analysis and visualization routines to work with the visibility data. These core routines *do not* have a `casatools` dependency, and so they should have maximal compatability with current Python versions. They are available via `pip install visread`.

If you are interested in working with the visread package, we recommend reading the [Installation Guide](installation.md) to learn more about suggested install patterns.

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

tutorials/introduction_to_casatools
tutorials/rescale_AS209_weights
casatools-api.rst
installation.rst
api.rst
api-casa-dep.rst
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
