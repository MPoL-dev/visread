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
copyright = "2021, Ian Czekala"
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
]

autodoc_mock_imports = ["numpy", "casatools"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# -- Options for Read the Docs - Github Action-------------------
# https://rtds-action.readthedocs.io/en/latest/index.html

# RTDs-action
if "GITHUB_TOKEN" in os.environ:
    print("RUNNING RTDS ACTION")
    extensions.append("rtds_action")

    rtds_action_github_repo = "MPoL-dev/visread"
    # The path where the artifact should be extracted
    # Note: this is relative to the conf.py file!
    rtds_action_path = "tutorials"
    # The "prefix" used in the `upload-artifact` step of the action
    rtds_action_artifact_prefix = "notebooks-for-"
    rtds_action_github_token = os.environ["GITHUB_TOKEN"]
