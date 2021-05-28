.. visread documentation master file, created by
   sphinx-quickstart on Wed Jan  6 11:59:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Visread documentation
=====================

|Tests badge|

.. |Tests badge| image:: https://github.com/MPoL-dev/visread/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/MPoL-dev/visread/actions/workflows/tests.yml

**What is this package**? This package is mostly a collection of documentation and tutorials demonstrating how to read visibilities (and associated metadata) from a calibrated CASA Measurement Set file into memory in a Python program using the built in CASA tools like `table` and `ms`. Once in memory, you can analyze them using your existing Python software stack, or save them to a binary file format (like ``*.npz`` or ``*.hdf5``) so that you can transport them to a computing environment free from a CASA dependency. 

**To get started**, you can get some ideas for working with `table` and `ms` directly from the *Introduction to CASA tools*. For convenience, you can install the *visread* package to provide some lightweight routines for common visibility manipulations, which are built on top of the core `casatools` functionality. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials/introduction_to_casatools
   installation.rst
   tutorials/plot_baselines
   api.rst
   introduction_to_spectral_lines.rst

Citation
========

If you find this piece of code useful, please cite::

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
