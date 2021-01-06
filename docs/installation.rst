Installation 
============

Because this package relies on the CASA 6.x modular installation, it is unfortunately subject to the stringent package requirements currently imposed by that environment. As of January 2021, these are ``Python=3.6`` and ``libgfortran3``. This means that CASA6 and by extension this package **will not work on Python 3.7, 3.8, or 3.9**. More information on these requirements is `here <https://casa.nrao.edu/casadocs-devel/stable/usingcasa/obtaining-and-installing>`_. Beyond this CASA dependency, the `visread` package itself is not tied to Python=3.6, so as the CASA requirements advance, so too will this package.

Following standard practice, it's recommended that you first create and activate a `Python virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ specific to your current project, whatever that may be. The CASA docs explicitly recommend using ``venv`` instead of a `conda environment <https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html>`_ (or a ``virtualenv``), though you may find success in some instances. 

In the following example of a virtual environment, the first line uses the ``venv`` tool to create a subdirectory named ``venv``, which hosts the virtual environment files. The second line activates this environment in your current shell session. Keep in mind that this is something you'll need to do for every shell session, since the point of virtual environments is to create an environment specific to your project.::

    $ python3 -m venv venv
    $ source venv/bin/activate

Then you can install ``visread`` and the CASA dependencies with ::

    $ pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple visread

Development
-----------

If you're interested in extending the package, you can install it locally for development::

    $ pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple -e .[dev]

If you make a useful change, please consider submitting a `pull request <https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request>`_ to the `github repo <https://github.com/MPoL-dev/visread>`_!