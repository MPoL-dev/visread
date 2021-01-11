# visread

![Python package](https://github.com/MPoL-dev/visread/workflows/Python%20package/badge.svg) [![Documentation Status](https://readthedocs.org/projects/visread/badge/?version=latest)](https://visread.readthedocs.io/en/latest/?badge=latest)

A minimal tool built on CASA6 to read visibilities directly from calibrated CASA Measurement Sets

# Introduction

The purpose of this package is to provide a quick and easy `read("myFile.ms")` function to load visibilities from a calibrated CASA Measurement Set file into memory in a Python program. You may wish to plot the baseline distributions, analyze the visibilities themselves, or synthesize images from them using software like [MPoL](https://github.com/MPoL-dev).

The visibilities are returned as an object. 

Code Example 

    import visread
    vis = visread.read("myFile.ms")
    print(vis)

More information is provided in the Documentation.

Currently, there is only one primary visibility container for spectral line measurement sets, whose attributes are driven by the development needs of the MPoL project. A set of single-channel continuum visibilities is just a subset of this.

If you identify and configure an additional visibility container (i.e., polarization), pull requests are welcome.

# Installation

Because this package relies on CASA6, it is unfortunately subject to the stringent package requirements currently imposed by the modular CASA environment. As of January 2021, these are Python=3.6 and `libgfortran3`. This means that CASA6 and by extension this package **will not work on Python 3.7, 3.8, or 3.9**. More information on these requirements is [here](https://casa.nrao.edu/casadocs-devel/stable/usingcasa/obtaining-and-installing). Beyond this CASA dependency, the `visread` package itself is not tied to Python=3.6, so as the CASA requirements advance, so too will this package.

Following standard practice, it's recommended that you first create and activate a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) specific to your current project whatever it may be. The CASA docs explicitly recommend using `venv` instead of a [conda environment](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html) (or a `virtualenv`), though you may find success in some instances. 

In the following example of a virtual environment, the first line uses the `venv` tool to create a subdirectory named `venv`, which hosts the virtual environment files. The second line activates this environment in your current shell session. Keep in mind that this is something you'll need to do for every shell session, since the point of virtual environments is to create an environment specific to your project.

    $ python3 -m venv venv
    $ source venv/bin/activate

Then you can install 

    $ pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple visread

## Development

If you're interested in extending the package, you can install it locally for development

    $ pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple -e .[dev]

If you make a useful change, please consider submitting a [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) to the [github repo](https://github.com/MPoL-dev/visread)!

# Lineage

The code in this package is fairly simple, but it is also built using CASA knowledge gained from a number of collaborators, who are credited in the CONTRIBUTORS.md file. You may also be interested in investigating the [`vis_sample`](https://github.com/AstroChem/vis_sample) and [`UVHDF5`](https://github.com/AstroChem/UVHDF5) packages, as they provide some similar capabilities with additional functionality.


