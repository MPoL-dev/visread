(installation-label)=
# Visread Installation

:::{note}
This page describes how to install the *visread* package, along with the CASA 6.x modular installation. Note that you *do not* need to install visread or any other packages to read visibilities from CASA. All of the routines in the *Introduction to CASA tools* will work directly with the tools [built in to the monolithic CASA distributions](https://casadocs.readthedocs.io/en/stable/api/casatools.html).
:::

:::{note}
If you are using an operating system that doesn't support the installation of the modular CASA 6.x distribution, you can still use the casatools directly (e.g., `tb` and `ms`) via the interactive prompt of the 'monolithic' CASA 5.x series. You just won't be able to install the wrapper functionality of *visread* (though you are welcome to copy and paste relevant source code to the terminal).
:::

## Installation

Following standard practice, it's recommended that you first create and activate a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) specific to your current project, whatever that may be. The CASA docs explicitly recommend using `venv` instead of a [conda environment](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html) (or a `virtualenv`), though it's possible that the other environments might still work.

In the following example of a virtual environment, the first line uses the `venv` tool to create a subdirectory named `venv`, which hosts the virtual environment files. The second line activates this environment in your current shell session. Keep in mind that this is something you'll need to do for every shell session, since the point of virtual environments is to create an environment specific to your project

```
python3 -m venv venv
source venv/bin/activate
```

Then you can install *visread* and the CASA dependencies based on your expected usage pattern.


## Usage Patterns

Consider which of the two patterns matches your situation.

### Pattern 1

You are unable to install Modular CASA (e.g., `casatools`) into your primary computing environment. Common reasons include incompatible Python versions (e.g., you are running Python 3.12, but Modular CASA only installs into Python 3.8) or operating systems (e.g., you are running MacOS 14, but Modular CASA only installs into MacOS 12).

So, you normally work with CASA to reduce your data in a specialized environment that supports the installation of CASA. Presumably there are factors that make this environment more difficult to access than your primary environment (such as SSH or VNC to a server), otherwise it would probably be your primary environment.

We suggest the following workflow. In your specialized, CASA-friendly environment, install 

```
pip install 'visread[casa]'
```

and use the visread casa-based routines to help export the visibilities from the measurement set to a common data format, like `.npy` or `.asdf`. Then, transfer this data file to your primary environment.

In your primary environment, install 

```
pip install visread
```

(without the CASA dependency), so that you can use the data visualization features of visread.

### Pattern 2

You are able to install Modular CASA into your primary computing environment. In this case, simply install 

```
pip install 'visread[casa]'
```

directly into your primary environment and work on everything in the same place.


If you have any problems with installation, please file a detailed [Github issue](https://github.com/MPoL-dev/visread/issues).


## Development

If you're interested in extending the package, you can clone it from GitHub and then install it locally for development:

```
$ pip install -e '.[dev]'
```

If you make a change that you think will be useful, please consider submitting a [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) to the [github repo](https://github.com/MPoL-dev/visread)!

## Testing

The test suite contains two sets of tests. The first "core" set does not require `casatools` or `casatasks`, and can be installed via 

```
pip install -e '.[test]'
```

You can run the tests via

```
python -m pytest
```

The test suite has logic to determine if `casatools` and `casatasks` are installed on your system. If these packages are not installed, then the test suite will simply skip the tests that require these packages.

If you have an environment in which you are able to install the casa dependencies, then do

```
pip install -e '.[test,casa]'
```

and run 

```
python -m pytest
```

Now, all tests in the suite should run. If any tests fail on your machine, please file a detailed [Github issue](https://github.com/MPoL-dev/visread/issues).

### Viewing test and debug plots

Some tests produce temporary files, like plots, that could be useful to view for development or debugging. Normally these are saved to a temporary directory created by the system which will be cleaned up after the tests finish. To preserve them, first create a plot directory (e.g., `plotsdir`) and then run the tests with this `--basetemp` specified

```
mkdir plotsdir
python -m pytest --basetemp=plotsdir
```
