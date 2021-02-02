# Testing for visread

Testing is carried out with `pytest`. You can install the package dependencies for testing via

    $ pip install .[test]

after you've cloned the repository and changed to the root of the repository (`visread`). This installs the extra packages required for testing (they are listed in `setup.py`).

To run all of the tests, from the root of the repository, invoke

    $ python -m pytest


The tests create a temporary directory to save all testing output, which is deleted after the tests run. If you'd like to capture some of the output from the tests (for example plots to look at), you can first create an output directory, like ``plotsdir`` for example, and specify this in your invocation

    $ python -m pytest --basetemp=plotsdir