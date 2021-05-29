# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# %matplotlib inline

# + nbsphinx="hidden"
# %run notebook_setup
# -

# # Walkthrough: Examining DSHARP AS 209 Weights and Exporting Visibilities
#
# In this walkthrough tutorial, we'll use CASA tools to examine the visibilities and weights of a real multi-configuration dataset from the DSHARP survey.
# 
# ## Downloading the calibrated measurement set.
# 
# The full datasets from the DSHARP data release are available [online](https://almascience.eso.org/almadata/lp/DSHARP/), and the full description of the survey is provided in [Andrews et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..41A/abstract). 
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.utils.data import download_file
import tempfile
import tarfile
import os

# START uncomment to use pre-cleaned ms locally 

# load the mock dataset of the ALMA logo
fname_tar = download_file(
    "https://almascience.eso.org/almadata/lp/DSHARP/MSfiles/AS209_continuum.ms.tgz",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)

# extract the measurement set to a local directory
temp_dir = tempfile.TemporaryDirectory()
curdir = os.getcwd()
os.chdir(temp_dir.name)

with tarfile.open(fname_tar) as tar:
    tar.extractall()

!ls

# END uncomment

fname = "AS209_continuum.ms"

# Let's import and then instantiate the relevant CASA tools, [table](https://casa.nrao.edu/casadocs-devel/stable/global-tool-list/tool_table/methods) and [ms](https://casa.nrao.edu/casadocs-devel/stable/global-tool-list/tool_ms/methods).

import casatools

tb = casatools.table()
ms = casatools.ms()

# Before you dive into the full analysis with CASA tools, it's a very good idea to inspect the measurement set using [listobs](https://casa.nrao.edu/casadocs-devel/stable/global-task-list/task_listobs/about). 
# 
# After you've done that, let's start exploring the visibility values. We can get the indexes of the unique spectral windows, which are typically indexed by the ``DATA_DESC_ID``

tb.open(fname + "/DATA_DESCRIPTION")
SPECTRAL_WINDOW_ID = tb.getcol("SPECTRAL_WINDOW_ID")
tb.close()
print(SPECTRAL_WINDOW_ID)

# We see that there are 25 separate spectral windows! This is because the DSHARP images were produced using all available Band 6 continuum data on each source---not just the long baseline observations acquired in ALMA cycle 4. The merging of all of these individual observations is what creates this structure with so many spectral windows. 

# Next, let's open the main table of the measurement set and inspect the column names

tb.open(fname)
colnames = tb.colnames()
tb.close()
print(colnames)

# Because there are multiple spectral windows which do not share the same dimensions, we cannot use the ``tb`` tool to read the data directly. If we try, we'll get an error.

try:
    tb.open(fname)
    weight = tb.getcol("WEIGHT")  # array of float64 with shape [npol, nvis]
    flag = tb.getcol("FLAG")  # array of bool with shape [npol, nchan, nvis]
    data = tb.getcol("DATA")  # array of complex128 with shape [npol, nchan, nvis]
except RuntimeError:
    print("We can't use table tools here... the spws have different numbers of channels")
finally:
    tb.close()

# So, we'll need to use the ``ms`` tool to read the visibilities for each spectral window, like so

ms.open(fname)
# select the spectral window
ms.selectinit(datadescid=0)
# query the desired columnames as a list
query = ms.getdata(["WEIGHT", "UVW", "DATA"])
# always a good idea to reset the earmarked data
ms.selectinit(reset=True)
ms.close()

# The returned query is a dictionary whose
# keys are the lowercase column names
print(query.keys())


ms.open(fname)
# select the spectral window
ms.selectinit(datadescid=0)
# query the desired columnames as a list
query = ms.getdata(["MODEL_DATA"])
# always a good idea to reset the earmarked data
ms.selectinit(reset=True)
ms.close()


# ## Using CASA tclean to produce MODEL_DATA
# 
# The full reduction scripts are [available online](https://almascience.eso.org/almadata/lp/DSHARP/scripts/AS209_continuum.py). Here we just reproduce the relevant ``tclean`` commands used to produce a FITS image from the final, calibrated measurement set.

if len(query["model_data"]) == 0:
    # We noticed that this dataset does not contain a MODEL_DATA column. To inspect the data, we'd like to make this.
    # Note that this process may take about 30 - 45 minutes, depending on your computing environment.
    print("empty model_data, CLEANing")
    # reproduce the DSHARP image using casa6
    import casatasks
    import shutil

    """ Define simple masks and clean scales for imaging """
    mask_pa  = 86  	# position angle of mask in degrees
    mask_maj = 1.3 	# semimajor axis of mask in arcsec
    mask_min = 1.1 	# semiminor axis of mask in arcsec
    mask_ra  = '16h49m15.29s'
    mask_dec = '-14.22.09.04'

    common_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
            (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)

    imagename = "AS209"

    # clear any existing image products
    for ext in [
        ".image",
        ".mask",
        ".model",
        ".pb",
        ".psf",
        ".residual",
        ".sumwt",
        ".image.pbcor",
    ]:
        obj = imagename + ext
        if os.path.exists(obj):
            shutil.rmtree(obj)


    casatasks.delmod(vis=fname)

    casatasks.tclean(vis=fname,
        imagename = imagename,
        specmode = 'mfs',
        deconvolver = 'multiscale',
        scales = [0, 5, 30, 100, 200],
        weighting='briggs',
        robust = -0.5,
        gain = 0.2,
        imsize = 3000,
        cell = '.003arcsec',
        niter = 50000,
        threshold = "0.08mJy",
        cycleniter = 300,
        cyclefactor = 1,
        uvtaper = ['.037arcsec','.01arcsec','162deg'],
        mask = common_mask,
        nterms = 1,
        savemodel="modelcolumn")

    if os.path.exists(imagename + '.fits'):
        os.remove(imagename + '.fits')
    casatasks.exportfits(imagename + '.image', imagename + '.fits', dropdeg=True, dropstokes=True)

    !ls


# ### Visualizing the CLEANed image

from astropy.io import fits 

hdul = fits.open("AS209.fits")
header = hdul[0].header
data = 1e3 * hdul[0].data  # mJy/pixel
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
norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(data))

fig, ax = plt.subplots(nrows=1, figsize=(4.5, 3.5))
fig.subplots_adjust(left=0.2, bottom=0.2)
im = ax.imshow(data, extent=ext, origin="lower", animated=True, norm=norm)
cb = plt.colorbar(im, label="mJy / pixel")
r = 2.2
ax.set_xlim(r, -r)
ax.set_ylim(-r, r)
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")


# ## Using the tclean model to calculate residual scatter
# now try getting the MODEL_DATA

ms.open(fname)
# select the spectral window
ms.selectinit(datadescid=0)
# query the desired columnames as a list
query = ms.getdata(["MODEL_DATA"])
# always a good idea to reset the earmarked data
ms.selectinit(reset=True)
ms.close()

print(query["model_data"])

# Let's calculate the residuals for each polarization (XX, YY) in units of $\sigma$, where
#
# $$
# \sigma = \mathrm{sigma\_rescale} \times \sigma_0
# $$
# 
# and 
#
# $$ 
# \sigma_0 = \sqrt{1/w}
# $$
# 
# The scatter is defined as
# 
# $$
# \mathrm{scatter} = \frac{\mathrm{DATA} - \mathrm{MODEL\_DATA}}{\sigma}
# $$
# We can turn this into a routine 



# ### Helper functions for examining weight scatter
# In this section, we'll define several functions to help us examine and plot the residual scatter. The commands in this document are only dependent on the CASA tools ``tb`` and ``ms``. But, if you find yourself using these routines frequently, you might consider installing the *visread* package, since similar commands are provided in the API.

def get_scatter_datadescid(datadescid, sigma_rescale=1.0, apply_flags=True):

    ms.open(fname)
    # select the key
    ms.selectinit(datadescid=datadescid)
    query = ms.getdata(["DATA", "MODEL_DATA", "WEIGHT", "UVW", "ANTENNA1", "ANTENNA2", "FLAG"])
    ms.selectinit(reset=True)
    ms.close()

    data, model_data, weight, flag = (
        query["data"],
        query["model_data"],
        query["weight"],
        query["flag"],
    )

    assert (
        len(model_data) > 0
    ), "MODEL_DATA column empty, retry tclean with savemodel='modelcolumn'"

    # subtract model from data
    residuals = data - model_data

    # calculate sigma from weight
    sigma = np.sqrt(1 / weight)
    sigma *= sigma_rescale

    # divide by weight, augmented for channel dim
    scatter = residuals / sigma[:, np.newaxis, :]

    # separate polarizations
    scatter_XX, scatter_YY = scatter
    flag_XX, flag_YY = flag

    if apply_flags:
        # flatten across channels
        scatter_XX = scatter_XX[~flag_XX]
        scatter_YY = scatter_YY[~flag_YY]

    return scatter_XX, scatter_YY

# Let's also write a function to plot a histogram, with a Gaussian 

def gaussian(x):
    r"""
    Evaluate a reference Gaussian as a function of :math:`x`

    Args:
        x (float): location to evaluate Gaussian

    The Gaussian is defined as

    .. math::

        f(x) = \frac{1}{\sqrt{2 \pi}} \exp \left ( -\frac{x^2}{2}\right )

    Returns:
        Gaussian function evaluated at :math:`x`
    """
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)

def scatter_hist(scatter_XX, scatter_YY, log=False, **kwargs):
    """
    Args:
        scatter_XX (1D numpy array)
        scatter_YY (1D numpy array)

    Returns:
        matplotlib figure
    """
    xs = np.linspace(-5, 5)

    figsize = kwargs.get("figsize", (6, 6))
    bins = kwargs.get("bins", 40)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=figsize)
    ax[0, 0].hist(scatter_XX.real, bins=bins, density=True, log=log)
    ax[0, 0].set_xlabel(
        r"$\Re \{ V_\mathrm{XX} - \bar{V}_\mathrm{XX} \} / \sigma_\mathrm{XX}$"
    )
    ax[0, 1].hist(scatter_XX.imag, bins=bins, density=True, log=log)
    ax[0, 1].set_xlabel(
        r"$\Im \{ V_\mathrm{XX} - \bar{V}_\mathrm{XX} \} / \sigma_\mathrm{XX}$"
    )

    ax[1, 0].hist(scatter_YY.real, bins=bins, density=True, log=log)
    ax[1, 0].set_xlabel(
        r"$\Re \{ V_\mathrm{YY} - \bar{V}_\mathrm{YY} \} / \sigma_\mathrm{YY}$"
    )
    ax[1, 1].hist(scatter_YY.imag, bins=bins, density=True, log=log)
    ax[1, 1].set_xlabel(
        r"$\Im \{ V_\mathrm{YY} - \bar{V}_\mathrm{YY} \} / \sigma_\mathrm{YY}$"
    )

    for a in ax.flatten():
        a.plot(xs, gaussian(xs))

    fig.subplots_adjust(hspace=0.25, top=0.95)

    return fig


def plot_histogram_datadescid(datadescid, sigma_rescale=1.0, log=False, apply_flags=True):

    scatter_XX, scatter_YY = get_scatter_datadescid(datadescid=datadescid, sigma_rescale=sigma_rescale, apply_flags=apply_flags)

    scatter_XX = scatter_XX.flatten()
    scatter_YY = scatter_YY.flatten()

    fig = scatter_hist(scatter_XX, scatter_YY, log=log)
    fig.suptitle("DATA_DESC_ID: {:}".format(datadescid))

# ## Checking scatter for each spectral window

# 7 no outliers, scaled fine
plot_histogram_datadescid(7, apply_flags=False)

# ### Visibility Outliers
# Note that we do need to apply the flags correctly. These outlier visibilities might adversely affect the image.

# 22 has outliers
plot_histogram_datadescid(22, apply_flags=False)

# Something looks a little bit strange compared to the previous spectral window. Let's try plotting things on a log scale to get a closer look.

plot_histogram_datadescid(22, apply_flags=False, log=True)

# It appears as though there are several "outlier" visibilities included in this dataset. If the calibration and data preparation went correctly, most likely, these visibilities are actually flagged. Let's try plotting only the valid, unflagged, visibilities  

plot_histogram_datadescid(22, apply_flags=True, log=True)

# Great, it looks like everything checks out.

# ### Incorrectly scaled weights

# 24 no outliers, but scaled incorrectly
plot_histogram_datadescid(24)

# That's strange, the scatter of these visibilities looks reasonably Gaussian, but the scatter is too large relative to what should be expected given the weight values. 
#
# If we rescale the $\sigma$ values to make them a factor of $\sqrt{2}$ larger (decrease the weight values by a factor of 2), it looks like everything checks out

plot_histogram_datadescid(24, sigma_rescale=np.sqrt(2))



# ## Rescaling weights for export.
# We can use the previous routines to iterate through plots of each spectral window. We see that the visibilities in the following spectral windows need to be rescaled by a factor of $\sqrt{2}$: 

SPWS_RESCALE = [9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# for ID in SPECTRAL_WINDOW_ID:
#     plot_histogram_datadescid(ID)

# 
#
# We'll draw upon the "Introduction to CASA tools" tutorial to read all of the visibilities, average polarizations, convert baselines to kilolambda, etc. The difference is that in this application we will need to treat the visibilities on a per-spectral window basis *and* we will need to rescale the weights when they are incorrect relative to the actual scatter.
