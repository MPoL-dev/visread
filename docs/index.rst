.. visread documentation master file, created by
   sphinx-quickstart on Wed Jan  6 11:59:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Visread documentation
=====================

*visread* provides a simple ``read("myFile.ms")`` function to load visibilities from a calibrated CASA Measurement Set file into memory in a Python program. You may wish to plot the baseline distributions, analyze the visibilities themselves, or synthesize images from them using software like `MPoL <https://github.com/MPoL-dev>`_.

Usage::

   import visread

   vis = visread.read(filename="myMeasurementSet.ms")
   # access your data with
   vis.frequencies  # frequencies in GHz
   vis.uu  # East-West spatial frequencies in klambda
   vis.vv  # North-South spatial frequencies in klambda
   vis.data_re  # real components of visibilities in Jy
   vis.data_im  # imaginary components of visibilities in Jy


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction.rst
   installation.rst
   api.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
