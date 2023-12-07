import pytest
import os
import numpy as np
from astropy.utils.data import download_file

# ascertain if casatasks is installed
try:
    import casatasks

    no_casa = False
except ModuleNotFoundError:
    no_casa = True

# some fake data, model, weights
@pytest.fixture
def data_dict():
    """
    Create a mock dataset that closely similates what we might get from a CASA measurement set.

    This has two polarizations, XX and YY.
    This has more than one channel, in this case 8.
    And many baselines.
    """
    npol = 2
    nchan = 8
    nvis = 5000

    # baselines are the same for all polarizations and channels
    uu = np.random.uniform(-1000, 1000, size=nvis) # meters
    vv = np.random.uniform(-1000, 1000, size=nvis) # meters

    freq = np.linspace(230.0e9, 231.0e9, num=nchan) # Hz

    model = np.ones((npol, nchan, nvis)) + np.zeros((npol, nchan, nvis)) * 1.0j

    # assumed to be the same for all channels, like basic CASA
    sigma = 0.2 * np.ones((npol, nvis))
    weight = 1 / sigma**2

    # temporarily broadcast sigma to the full size for the noise call
    noise_real = np.random.normal(scale=sigma[:,np.newaxis,:], size=(npol, nchan, nvis))
    noise_imag = np.random.normal(scale=sigma[:,np.newaxis,:], size=(npol, nchan, nvis))
    noise = noise_real + noise_imag * 1.0j
    data = model + noise

    # no data flagged here
    flag = np.zeros((npol, nchan, nvis), dtype=np.bool_)

    return {"uu":uu, "vv":vv, "freq":freq, "data":data, "flag":flag, "model":model, "weight":weight}


# create a fixture for temporary MS file itself
@pytest.mark.skipif(no_casa, reason="modular casa not available on this system")
@pytest.fixture(scope="session")
def ms_cont_path(tmp_path_factory):
    # Generate a continuum MS

    # load test FITS file provided by package
    # assuming we're running from top directory
    fits_path = "tests/logo_cont.fits"
    fits_path = download_file(
        "https://zenodo.org/record/4711811/files/logo_cont.fits",
        cache=True,
        pkgname="visread",
    )

    # use the tmp_path directory provided by pytest to automatically
    # clean up all of the files when we're done
    outdir = str(tmp_path_factory.mktemp("cont"))

    inbright = 2e-5  # Jy/pixel
    indirection = "J2000 04h55m10.98834s +030.21.58.879285"

    # change to outdir
    curdir = os.getcwd()
    os.chdir(outdir)

    casatasks.simobserve(
        skymodel=fits_path,
        inbright="{:}Jy/pixel".format(inbright),
        incenter="230GHz",
        indirection=indirection,
        inwidth="8GHz",
        hourangle="transit",
        totaltime="3600s",
        graphics="none",
        overwrite=True,
        obsmode="int",  # interferometer
        antennalist=curdir + "/" + "tests/alma.cycle7.7.cfg",
    )

    os.chdir(curdir)
    ms_path = outdir + "/sim/sim.alma.cycle7.7.ms"

    return ms_path


@pytest.mark.skipif(no_casa, reason="modular casa not available on this system")
@pytest.fixture(scope="session")
def ms_cube_path(tmp_path_factory):
    # Generate an MS for a spectral cube

    # load test FITS file provided by package
    # assuming we're running from top directory
    fits_path = download_file(
        "https://zenodo.org/record/4711811/files/logo_cube.fits",
        cache=True,
        pkgname="visread",
    )

    # use the tmp_path directory provided by pytest to automatically
    # clean up all of the files when we're done
    outdir = str(tmp_path_factory.mktemp("cube"))

    # change to outdir
    curdir = os.getcwd()
    os.chdir(outdir)

    casatasks.simobserve(
        skymodel=fits_path,
        hourangle="transit",
        totaltime="3600s",
        graphics="none",
        thermalnoise="tsys-atm",
        overwrite=True,
        obsmode="int",  # interferometer
        antennalist=curdir + "/" + "tests/alma.cycle7.7.cfg",
    )

    os.chdir(curdir)
    ms_path = outdir + "/sim/sim.alma.cycle7.7.ms"

    # path to MS cube with MODEL_DATA column
    return ms_path
