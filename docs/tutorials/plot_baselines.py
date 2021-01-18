# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Quickstart
#
# This is a short example designed to show you how to use *visread* to extract the visibility measurements from a measurement set.

# ## Generate a mock measurement set
#
# In reality, you'll just want to jump right in with your own measurement set containing your data. For this tutorial, though, we'll first create a mock dataset using CASA's *simobserve* task.
#
# As a starting point, we'll use a mock sky brightness distribution of the ALMA logo (included in the package). Just to orient ourselves, let's take a look at it first.

from astropy.io import fits

import matplotlib.pyplot as plt
import numpy as np


hdul = fits.open("../../tests/logo_cube.fits")
header = hdul[0].header
data = hdul[0].data
# get the coordinate labels
nx = header["NAXIS1"]
ny = header["NAXIS2"]
# RA coordinates
CDELT1 = 3600 * header["CDELT1"]  # arcsec (converted from decimal deg)
CRPIX1 = header["CRPIX1"] - 1.0  # Now indexed from 0
# DEC coordinates
CDELT2 = 3600 * header["CDELT2"]  # arcsec
CRPIX2 = header["CRPIX2"] - 1.0  # Now indexed from 0
RA = (np.arange(nx) - nx / 2) * CDELT1  # [arcsec]
DEC = (np.arange(ny) - ny / 2) * CDELT2  # [arcsec]
# extent needs to include extra half-pixels.
# RA, DEC are pixel centers
ext = (
    RA[0] - CDELT1 / 2,
    RA[-1] + CDELT1 / 2,
    DEC[0] - CDELT2 / 2,
    DEC[-1] + CDELT2 / 2,
)  # [arcsec]
freqs = header["CRVAL3"] + np.arange(header["NAXIS3"]) * header["CDELT3"]  # [Hz]
nchan = len(data)

fig, ax = plt.subplots(nrows=1, ncols=nchan)
for i in range(nchan):
    ax[i].imshow(data[i], extent=ext, origin="lower")
    ax[i].set_title(r"{:.3f} GHz".format(freqs[i] * 1e-9))
ax[0].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[0].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
for i in range(1, nchan):
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])
plt.show()
