---
file_format: mystnb
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```{code-cell} ipython3 
:tags: [hide-cell]
%run notebook_setup
```

(intro-casatools-label)=

# Introduction to CASA tools

This tutorial is meant to provide an introduction to working with some the 'tools' provided in the [CASA package](https://casadocs.readthedocs.io/en/stable/index.html). In particular, we'll focus on the `msmetadata`, `tb`, and `ms` tools. 

## Introduction to measurement sets
Before you begin, it's worthwhile reviewing the CASA documentation on the measurement set, which is the default storage format for radio interferometric observations. The basics are [here](https://casadocs.readthedocs.io/en/stable/notebooks/casa-fundamentals.html#MeasurementSet-Basics). To make a long story short, a measurement set (e.g., ``my_ALMA_data.ms``) is a folder containing a set of binary 'tables' with your data and metadata. The contents within this measurement set folder serve as a [relational database](https://en.wikipedia.org/wiki/Relational_database). It helps to keep this structure in mind as we navigate its contents.

A good first task is to load up an interactive prompt of CASA and work through the [Data Examination](https://casadocs.readthedocs.io/en/stable/notebooks/data_examination.html) routines like ``listobs`` and ``plotms`` to familiarize yourself with the basic structure of your measurement set. 

It will be very helpful to know things like how many spectral windows there are, how many execution blocks there are, and what targets were observed. Hopefully you've already ``split`` out your data such that it only contains the target under consideration and does not include any extra calibrator targets, which can simplify many of the queries.  

CASA also provides the ``casabrowser/browsetable``tool, which is very handy for graphically exploring the structure and contents of this relational database. If something about the structure of the measurement set doesn't make sense, it's usually a good idea to open up ``browsetable`` and dig into the structure of the individual tables.


## Mock data set 

To experiment with reading and plotting visibilities, we'll use a measurement set that we prepared using simobserve. The full commands to generate the measurement set are available via the `mpoldatasets` package [here](https://github.com/MPoL-dev/mpoldatasets/blob/main/products/ALMA-logo/simobserve_cube.py), but you could just as easily use your own measurement set. You can download the measurement set using the following commands.

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c
from astropy.utils.data import download_file
import tempfile
import tarfile
import os
```

```{code-cell}
# load the mock dataset of the ALMA logo
fname_tar = download_file(
    "https://zenodo.org/record/4711811/files/logo_cube.noise.ms.tar.gz",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)
```

```{code-cell}
# extract the measurement set to a local directory
temp_dir = tempfile.TemporaryDirectory()
curdir = os.getcwd()
os.chdir(temp_dir.name)
```

```{code-cell}
with tarfile.open(fname_tar) as tar:
    tar.extractall()
```

Now we've successfully downloaded and extracted the measurement set

```{code-cell}
!ls 
```

```{code-cell}
fname = "logo_cube.noise.ms"
```

If you're working with your own measurement set, you can start from the next section.


## CASA tools setup
CASA provides a set of lower-level "tools" for direct interaction with the measurement set contents. The full API list is available [here](https://casadocs.readthedocs.io/en/stable/api/casatools.html). You can access the CASA tools from within your Python environment if you've successfully installed ``casatools`` package. If you are unable to install the modular package, you can always use the casatools directly inside of the monolithic CASA intepreter. If you're working directly within the CASA interpreter, you can skip this section and move directly to the example queries.

Let's import and then instantiate the relevant CASA tools, [msmetadata](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.msmetadata.html#casatools.msmetadata), [ms](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html), and [table](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.table.html).

```{code-cell}
import casatools
```

```{code-cell}
msmd = casatools.msmetadata()
ms = casatools.ms()
tb = casatools.table()
```

## Example metadata queries

### Column names of main table
Open the main table of the measurement set and inspect the column names

```{code-cell}
tb.open(fname)
colnames = tb.colnames()
tb.close()
print(colnames)
```

### Unique spectral window IDs
Get the indexes of the unique spectral windows, typically indexed by the ``DATA_DESC_ID``

```{code-cell}
msmd.open(fname)
spws = msmd.datadescids() 
msmd.done()
print(spws)
```

``DATA_DESC_ID`` might not always be a perfect stand-in for ``SPECTRAL_WINDOW_ID``, see the [data description table](https://casadocs.readthedocs.io/en/stable/notebooks/casa-fundamentals.html#MeasurementSet-Basics) for more information. You could also try accessing the ``DATA_DESCRIPTION`` table directly by providing it as a subdirectory to the main table like so

```{code-cell}
tb.open(fname + "/DATA_DESCRIPTION")
SPECTRAL_WINDOW_ID = tb.getcol("SPECTRAL_WINDOW_ID")
tb.close()
print(SPECTRAL_WINDOW_ID)
```

### How many channels in each spw
Let's figure out how many channels there are, and what their frequencies are (in Hz)

```{code-cell}
spw_id = 0
msmd.open(fname)
chan_freq = msmd.chanfreqs(spw_id)
msmd.done()
nchan = len(chan_freq)
print(nchan)
print(chan_freq)
```


## Example ms tool queries
Sometimes your measurement set might contain multiple spectral windows with different numbers of channels or polarizations. In such a situation, the above queries with the ``tb`` tool will most likely fail, because they are trying to read data with inherently different shapes into an array with one common shape. One solution is to use the [ms](https://casa.nrao.edu/casadocs-devel/stable/global-tool-list/tool_ms/methods) tool.

Two useful methods are ``ms.selectinit`` followed by ``ms.getdata``. The first earmarks a subset of data based upon the given ``DATA_DESC_ID``, the second returns the actual data from the columns specified. It is possible to retrieve the data from a single spectral window because all data within a given spectral window will have the same polarization and channelization setup. The downside is that if your dataset has multiple spectral windows, you will need to repeat the queries for each one. It isn't obvious that there is a faster way to retrieve the data in such a situation, however.

```{code-cell}
ms.open(fname)
# select the spectral window
ms.selectinit(datadescid=0)
# query the desired columnames as a list
query = ms.getdata(["WEIGHT", "UVW"])
# always a good idea to reset the earmarked data
ms.selectinit(reset=True)
ms.close()
```

```{code-cell}
# The returned query is a dictionary whose
# keys are the lowercase column names
print(query)
```

```{code-cell}
# And the data values are the same as before
uvw = query["uvw"]
print(uvw.shape)
```


## Working with baselines

Baselines are stored in the `uvw` column, in units of meters. 

### Get the baselines
Read the baselines (in units of [m])

```{code-cell}
spw_id = 0
ms.open(fname)
ms.selectinit(spw_id)
d = ms.getdata(["uvw"])  
ms.done()
# d["uvw"] is an array of float64 with shape [3, nvis]
uu, vv, ww = d["uvw"]  # unpack into len nvis vectors
```

We can plot these up using matplotlib

```{code-cell}
fig, ax = plt.subplots(nrows=1, figsize=(3.5, 3.5))
ax.scatter(uu, vv, s=1.5, rasterized=True, linewidths=0.0, c="k")
ax.set_xlabel(r"$u$ [m]")
ax.set_ylabel(r"$v$ [m]")
```

### Convert baselines to units of lambda
To convert the baselines to units of lambda, we need to know the observing frequency (or wavelength). For a spectral window with multiple channels (like this one), the baselines (if expressed in units of lambda) will be slightly different for each channel because the observing frequency is different for each channel. Here is one way to go about broadcasting the baselines to all channels and then converting them to units of lambda.

```{code-cell}
# broadcast to the same shape as the data
# stub to broadcast uu,vv, and weights to all channels
broadcast = np.ones((nchan, 1))
uu = uu * broadcast
vv = vv * broadcast
```

```{code-cell}
# calculate wavelengths for each channel, in meters
wavelengths = c.value / chan_freq[:, np.newaxis]  # m
```

```{code-cell}
# convert baselines to lambda
uu = uu / wavelengths  # [lambda]
vv = vv / wavelengths  # [lambda]
```

Let's plot up the baseline coverage for the first channel of the cube (index `0`). For the plot only, we'll convert to units of k$\lambda$ to make it easier to read.

```{code-cell}
fig, ax = plt.subplots(nrows=1, figsize=(3.5, 3.5))
ax.scatter(uu[0] * 1e-3, vv[0] * 1e-3, s=1.5, rasterized=True, linewidths=0.0, c="k")
ax.set_xlabel(r"$u$ [k$\lambda$]")
ax.set_ylabel(r"$v$ [k$\lambda$]")
```

## Working with visibilities

### Read the visibility values and inspect flags
Visibilities are stored in the main table of the measurement set. As long as all spectral windows have the same number of channels and polarizations, it should be possible to read visibilities using the table tool. If not, try the `ms` tool.

```{code-cell}
tb.open(fname)
weight = tb.getcol("WEIGHT")  # array of float64 with shape [npol, nvis]
flag = tb.getcol("FLAG")  # array of bool with shape [npol, nchan, nvis]
data_raw = tb.getcol("DATA")  # array of complex128 with shape [npol, nchan, nvis]
data_corrected = tb.getcol(
    "CORRECTED_DATA"
)  # array of complex128 with shape [npol, nchan, nvis]
tb.close()
```

Depending on how you've calibrated and/or split out your data, your measurement set may or may not have a ``CORRECTED_DATA`` column. Usually, this is the one you want. If the column doesn't exist in your measurement set, try the ``DATA`` column instead.

For each visibility $i$ there is also an accompanying ``WEIGHT`` $w_i$, which should correspond to the statistical uncertainty on the real and imaginary component of each visibility. Formally, the weight is defined as

$$
w_i = \frac{1}{\sigma_i^2}
$$

and we expect that the noise is independent for each of the real and imaginary components of a visibility measurement. This means that we expect

$$
\Re \{V_i\} = \Re\{\mathcal{V}\}_i + \epsilon_1
$$

and

$$
\Im \{V_i\} = \Im\{\mathcal{V}\}_i + \epsilon_2
$$

where for each real and each imaginary component $\epsilon$ is a new noise realization from a mean-zero Gaussian with standard deviation $\sigma_i$, i.e.,

$$
\epsilon \sim \mathcal{N}\left ( 0, \sigma_i \right )
$$

CASA has a [history](https://casa.nrao.edu/casadocs-devel/stable/calibration-and-visibility-data/data-weights) of different weight definitions, so this is definitely an important column to pay attention to, especially if your data was not acquired and processed in very recent cycles.


A common ALMA observation mode is dual-polarization, XX and YY. If this is the case for your observations, the data array will contain an extra dimension for each polarization.

```{code-cell}
print(data_corrected.shape)
```

Since it has a complex type, the data array contains the real and imaginary values

```{code-cell}
print(data_corrected.real)
print(data_corrected.imag)
```

In the process of calibrating real-world data, some occasional visibilities need to be flagged or excluded from the analysis. The ``FLAG`` column is a boolean flag that is ``True`` if the visibility *should be excluded*. To check if any visibilities are flagged in this spectral window, we can just see if there are any ``True`` values in the ``FLAG`` column

```{code-cell}
print("Are there any flagged visibilities?", np.any(flag))
```

Since this is a simulated measurement set, this is not surprising. When you are using real data, however, this is an important thing to check. If you *did* have flagged visibilities, to access only the valid visibilities you would do something like the following

```{code-cell}
data_good = data_corrected[~flag]
```

where the ``~`` operator helps us index only the visibilities which *are not flagged*. Unfortunately, indexing en masse like this removes any polarization and channelization dimensionality, just giving you a flat list

```{code-cell}
print(data_good.shape)
```

so you'll need to be more strategic about this if you'd like to preserve the channel dimensions, perhaps by using a masked array or a ragged list.


### Read visibility values and averaging polarizations
If your dataset is dual-polarization but you are not interested in treating the polarizations separately, it might be worth it to average the polarization channels together.

```{code-cell}
tb.open(fname)
uvw = tb.getcol("UVW")  # array of float64 with shape [3, nvis]
weight = tb.getcol("WEIGHT")  # array of float64 with shape [npol, nvis]
flag = tb.getcol("FLAG")  # array of bool with shape [npol, nchan, nvis]
data = tb.getcol("CORRECTED_DATA")  # array of complex128 with shape [npol, nchan, nvis]
tb.close()
```

```{code-cell}
# average the polarizations
# https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
data = np.sum(data * weight[:, np.newaxis, :], axis=0) / np.sum(weight, axis=0)
```

```{code-cell}
# flag the data if either polarization was flagged
flag = np.any(flag, axis=0)
```

```{code-cell}
# combine the weights across polarizations
weight = np.sum(weight, axis=0)
```

```{code-cell}
# Calculate the "radial" baseline
uu, vv, ww = uvw
qq = np.sqrt(uu ** 2 + vv ** 2)
```

```{code-cell}
# calculate visibility amplitude and phase
amp = np.abs(data)
phase = np.angle(data)
```

```{code-cell}
# Let's plot up the visibilities for the first channel
fig, ax = plt.subplots(nrows=1, figsize=(3.5, 3.5))
ax.scatter(qq, amp[0], s=1.5, rasterized=True, linewidths=0.0, c="k")
ax.set_xlabel(r"$q$ [m]")
ax.set_ylabel(r"Amplitude [Jy]")
```


## Exporting visibilities
This slightly larger example shows how to combine some of the above queries to read channel frequencies, baselines, and visibilites, and export them from the CASA environment saved as an ``*.npz`` file. We use the `table` tool here, but you could also use the `ms` tool.

```{code-cell}
# query the data
tb.open(fname)
ant1 = tb.getcol("ANTENNA1")  # array of int with shape [nvis]
ant2 = tb.getcol("ANTENNA2")  # array of int with shape [nvis]
uvw = tb.getcol("UVW")  # array of float64 with shape [3, nvis]
weight = tb.getcol("WEIGHT")  # array of float64 with shape [npol, nvis]
flag = tb.getcol("FLAG")  # array of bool with shape [npol, nchan, nvis]
data = tb.getcol("CORRECTED_DATA")  # array of complex128 with shape [npol, nchan, nvis]
tb.close()
```

```{code-cell}
# get the channel information
tb.open(fname + "/SPECTRAL_WINDOW")
chan_freq = tb.getcol("CHAN_FREQ")
num_chan = tb.getcol("NUM_CHAN")
tb.close()
chan_freq = chan_freq.flatten()  # Hz
nchan = len(chan_freq)
```

Check to make sure the channels are in blushifted to redshifted order, otherwise reverse channel order

```{code-cell}
if (nchan > 1) and (chan_freq[1] > chan_freq[0]):
    # reverse channels
    chan_freq = chan_freq[::-1]
    data = data[:, ::-1, :]
    flag = flag[:, ::-1, :]
```

Keep only the cross-correlation visibilities and throw out the auto-correlation visibilities (i.e., where ``ant1 == ant2``)

```{code-cell}
xc = np.where(ant1 != ant2)[0]
data = data[:, :, xc]
flag = flag[:, :, xc]
uvw = uvw[:, xc]
weight = weight[:, xc]
```

```{code-cell}
# average the polarizations
data = np.sum(data * weight[:, np.newaxis, :], axis=0) / np.sum(weight, axis=0)
flag = np.any(flag, axis=0)
weight = np.sum(weight, axis=0)
```

```{code-cell}
# After this step, ``data`` should be shape ``(nchan, nvis)`` and weights should be shape ``(nvis,)``
print(data.shape)
print(flag.shape)
print(weight.shape)
```

```{code-cell}
# when indexed with mask, returns valid visibilities
mask = ~flag
```

```{code-cell}
# convert uu and vv to lambda
uu, vv, ww = uvw  # unpack into len nvis vectors
# broadcast to the same shape as the data
# stub to broadcast uu,vv, and weights to all channels
broadcast = np.ones((nchan, 1))
uu = uu * broadcast
vv = vv * broadcast
weight = weight * broadcast
```

```{code-cell}
# calculate wavelengths in meters
wavelengths = c.value / chan_freq[:, np.newaxis]  # m
```

```{code-cell}
# calculate baselines in lambda
uu = uu / wavelengths  # [lambda]
vv = vv / wavelengths  # [lambda]
```

```{code-cell}
frequencies = chan_freq * 1e-9  # [GHz]
```

Since we have a multi-channel dataset, we need to make a decision about how to treat flagged visibilities. If we wanted to export only unflagged visibilities, there is a chance that some channels will have different number of flagged visibilities, which means we can no longer represent the data as a nicely packed, contiguous, rectangular array.

 Here, we sidestep this issue by just exporting *all* of the data (including flagged, potentially erroneous, visibilities), but, we also include the ``mask`` used to identify the valid visibilities. This punts on a final decision about how to store unflagged visibilities---the user will need to make sure that they correctly apply the mask when they load the data in a new environment.

```{code-cell}
# save data to numpy file for later use
np.savez(
    "visibilities.npz",
    frequencies=frequencies,  # [GHz]
    uu=uu,  # [lambda]
    vv=vv,  # [lambda]
    weight=weight,  # [1/Jy^2]
    data=data,  # [Jy]
    mask=mask,  # [Bool]
)
```
