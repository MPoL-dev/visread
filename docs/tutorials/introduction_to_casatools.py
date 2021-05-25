# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# %matplotlib inline

# + nbsphinx="hidden"
# %run notebook_setup
# -

# # Introduction to Measurement Sets
# Before you begin, it's worthwhile reviewing the CASA documentation on measurement sets. The basics are [here](https://casa.nrao.edu/casadocs-devel/stable/casa-fundamentals/the-measurementset) and the description of the v2 format is [here](https://casa.nrao.edu/casadocs-devel/stable/casa-fundamentals/measurement-set). To make a long story short, a measurement set (e.g., ``my_ALMA_data.ms``) is a folder containing a set of binary 'tables' with your data and metadata. The contents within this measurement set folder serve as a [relational database](https://en.wikipedia.org/wiki/Relational_database). It helps to keep this structure in mind as we navigate its contents.
#
# CASA provides the [casabrowser/browsetable](https://casa.nrao.edu/casadocs-devel/stable/calibration-and-visibility-data/data-examination-and-editing/browse-a-table) tool, which is very handy for graphically exploring the structure and contents of this relational database. If something about the structure of the measurement set doesn't make sense, it's usually a good idea to open up *browsetable* and dig into the structure of the individual tables.

# # Introduction to casatools
# CASA provides a set of lower-level "tools" for direct interaction with the measurement set contents. The full API list is available [here](https://casa.nrao.edu/casadocs-devel/stable/global-tool-list).

# download logo_cube_noise and extract from https://zenodo.org/record/4711811#.YKzhspNKidY
# note that full simobserve commands available in the [mpoldatasets](https://github.com/MPoL-dev/mpoldatasets/tree/main/products/ALMA-logo) repository.

# Use the simobserve cube dataset

# * inspect column names
# * inspect num of spws
# * get number of channels
# * get frequencies
# * broadcast baselines and convert to klambda
# * inspect flags
# * query columns with tb.getcol (note multiple spws, multiple channels)
# * average polarizations and weights
# * query with ms.selectinit and ms.getdata
# * save data to numpy file for later use