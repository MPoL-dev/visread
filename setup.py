import setuptools
import os.path

with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


version = get_version("src/visread/__init__.py")

EXTRA_REQUIRES = {
    "test": ["pytest", "casatasks"],
    "docs": [
        "sphinx>=2.3.0",
        "numpy",
        "nbsphinx",
        "sphinx_material",
        "sphinx_copybutton",
        "sphinx_rtd_theme",
        "jupyter",
        "jupytext",
        "nbconvert",
        "astropy",
        "casatasks",
        "rtds-action",
    ],
}

EXTRA_REQUIRES["dev"] = (
    EXTRA_REQUIRES["test"] + EXTRA_REQUIRES["docs"] + ["pylint", "black"]
)


setuptools.setup(
    name="visread",
    version=version,
    author="Ian Czekala",
    author_email="iczekala@psu.edu",
    description="Use CASA6 to read visibilities from Measurement Sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MPoL-dev/visread",
    install_requires=["numpy", "casatools"],
    extras_require=EXTRA_REQUIRES,
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.6",  # CASA6 requirement
)