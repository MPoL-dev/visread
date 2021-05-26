.. visread documentation master file, created by
   sphinx-quickstart on Wed Jan  6 11:59:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Visread documentation
=====================

**What is this package**? This package is mostly a set of documentation and tutorials demonstrating how to read visibilities (and associated metadata) from a calibrated CASA Measurement Set file into memory in a Python program using `casatools` like `table` and `ms`. You may wish to plot the baseline distributions, analyze the visibilities themselves, or synthesize images from them using software like `MPoL <https://github.com/MPoL-dev>`_.

To get started, you don't even need to install the *visread* package itself, you can get some ideas directly from the *Introduction to casatools*. The *visread* package provides some lightweight routines for common visibility manipulations, and is built on top of the core `casatools` functionality. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction.rst
   installation.rst
   api.rst
   tutorials/plot_baselines

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
