Visread Installation 
====================

.. note::

    This page describes how to install the *visread* package, along with the CASA 6.x modular installation. Note that you *do not* need to install visread or any other packages to read visibilities from CASA. All of the routines in the *Introduction to CASA tools* will work directly with the tools `built in to the monolithic CASA distributions <https://casa.nrao.edu/casadocs-devel/stable/casa-fundamentals/tasks-and-tools/casa-tools>`__.

Because the *visread* package relies on the CASA 6.x modular installation, it is unfortunately subject to the stringent package requirements currently imposed by that environment. As of June 2021, these are ``Python=3.6``, ``libgfortran3``, and a linux operating system (most likely Red Hat or CentOS). This means that CASA6 and by extension this package **will not work on Python 3.7, 3.8, or 3.9**, **MacOS**, and may even give you trouble on modern **Ubuntu** distros because of the ``libfortran3`` dependency. More information on these requirements is `here <https://casa.nrao.edu/casadocs-devel/stable/usingcasa/obtaining-and-installing>`_. The `visread` package itself is not tied to Python=3.6, so as the CASA requirements relax (anticipated by winter 2021-22), so too will the requirements of this package. 

.. note::

    If you are using an operating system that doesn't support the installation of the modular CASA 6.x distribution, you can still use the casatools directly (e.g., ``tb`` and ``ms``) via the interactive prompt of the 'monolithic' CASA 5.x series. You just won't be able to install the wrapper functionality of *visread* (though you are welcome to copy and paste relevant source code to the terminal).

First, if you don't already have it, install the ``libgfortran3`` library using the package management tool for your system.

Then, following standard practice, it's recommended that you first create and activate a `Python virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ specific to your current project, whatever that may be. The CASA docs explicitly recommend using ``venv`` instead of a `conda environment <https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html>`_ (or a ``virtualenv``), though it's possible that the other environments might still work.

In the following example of a virtual environment, the first line uses the ``venv`` tool to create a subdirectory named ``venv``, which hosts the virtual environment files. The second line activates this environment in your current shell session. Keep in mind that this is something you'll need to do for every shell session, since the point of virtual environments is to create an environment specific to your project ::

    $ python3 -m venv venv
    $ source venv/bin/activate

Then you can install *visread* and the CASA dependencies with ::

    $ pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple visread

If you have any problems, please file a detailed `Github issue <https://github.com/MPoL-dev/visread/issues>`_.

Development
-----------

If you're interested in extending the package, you can install it locally for development::

    $ pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple -e .[dev]

If you make a useful change, please consider submitting a `pull request <https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request>`_ to the `github repo <https://github.com/MPoL-dev/visread>`_!

Testing
-------

You can run the tests on the package by first installing it with:: 

    $ pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple -e .[test]

(which installs the casatasks and pytest packages, you could also just additionally install those yourself, too). Then run::

    $ python -m pytest 

The tests work by creating a fake measurement set using `simobserve <https://casa.nrao.edu/casadocs-devel/stable/global-task-list/task_simobserve/about>`_ and then reading it with *visread*. If any tests fail on your machine, please file a detailed `Github issue <https://github.com/MPoL-dev/visread/issues>`_.
