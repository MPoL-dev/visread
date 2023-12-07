# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os

# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# useful snippets from https://github.com/exoplanet-dev/exoplanet/blob/main/docs/conf.py
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("visread").version
except DistributionNotFound:
    __version__ = "unknown version"

project = "visread"
copyright = "2021-23, Ian Czekala"
author = "Ian Czekala"

# The full version, including alpha/beta/rc tags
version = __version__
release = __version__


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_nb",
]

# add in additional files
source_suffix = {
    ".ipynb": "myst-nb",
    ".rst": "restructuredtext",
    ".myst": "myst-nb",
    ".md": "myst-nb",
}

myst_enable_extensions = ["dollarmath", "colon_fence", "amsmath"]

# CASA imports are mocked with Sphinx ``autodoc_mock_imports`` because the CASA install
# requires ``libgfortran`` to be installed and apparently that's difficult with RTD.
autodoc_mock_imports = ["casatools"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": None}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/MPoL-dev/visread",
    "use_repository_button": True,
}

nb_execution_mode = "cache"
nb_execution_timeout = -1

nb_execution_excludepatterns = ["**.ipynb_checkpoints"]
myst_heading_anchors = 3
