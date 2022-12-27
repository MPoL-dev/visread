import numpy as np 
import matplotlib.pyplot as plt
import casatools 
from . import utils, scatter, process

ms = casatools.ms()

def plot_baselines(filename, datadescid):

    ms.open(filename)
    ms.selectinit(datadescid=datadescid)
    q = ms.getdata(["uvw"])
    ms.selectinit(reset=True)
    ms.close()

    u, v, w = q["uvw"] * 1e-3  # [km]

    fig, ax = plt.subplots(nrows=1)
    ax.scatter(u, v, s=0.5)
    ax.set_title("DATA_DESC_ID: {:}".format(datadescid))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$u$ [km]")
    ax.set_ylabel(r"$v$ [km]")

    return fig


def plot_averaged_scatter(scatter, log=False, **kwargs):

    xs = np.linspace(-5, 5)

    figsize = kwargs.get("figsize", (9, 4))
    bins = kwargs.get("bins", 40)

    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    ax[0].hist(scatter.real, bins=bins, density=True, log=log)
    ax[0].set_xlabel(r"$\Re \{ V - \bar{V} \} / \sigma$")
    ax[1].hist(scatter.imag, bins=bins, density=True, log=log)
    ax[1].set_xlabel(r"$\Im \{ V - \bar{V} \} / \sigma$")

    for a in ax.flatten():
        a.plot(xs, utils.gaussian(xs))

    fig.subplots_adjust(hspace=0.25, top=0.95)

    return fig


def plot_scatter_datadescid(
    filename,
    datadescid,
    log=False,
    sigma_rescale=1.0,
    chan_slice=None,
    apply_flags=True,
):
    r"""
    Plot a set of histograms of the scatter of the residual visibilities for the real and
    imaginary components of the XX and YY polarizations.

    Args:
        filename (string): measurement set filename
        datadescid (int): the DATA_DESC_ID to be queried
        log (bool): should the histogram values be log scaled?
        sigma_rescale (int):  multiply the uncertainties by this factor
        chan_slice (slice): if not None, a slice object specifying the channels to subselect
        apply_flags (bool): calculate the scatter *after* the flags have been applied

    Returns:
        matplotlib figure with scatter histograms
    """

    if chan_slice is not None:
        print("apply_flags setting is ignored when chan_slice is not None")

        scatter_XX, scatter_YY = scatter.get_scatter_datadescid(
            filename, datadescid, sigma_rescale, apply_flags=False
        )

        scatter_XX = scatter_XX[chan_slice]
        scatter_YY = scatter_YY[chan_slice]

    else:
        scatter_XX, scatter_YY = scatter.get_scatter_datadescid(
            filename, datadescid, sigma_rescale, apply_flags=apply_flags
        )

    scatter_XX = scatter_XX.flatten()
    scatter_YY = scatter_YY.flatten()

    fig = _scatter_hist(scatter_XX, scatter_YY, log=log)
    fig.suptitle("DATA_DESC_ID: {:}".format(datadescid))

    return fig


def _scatter_hist(scatter_XX, scatter_YY, log=False, **kwargs):
    """
    Args:
        scatter_XX (1D numpy array)
        scatter_YY (1D numpy array)

    Returns:
        matplotlib figure
    """
    xs = np.linspace(-5, 5)

    figsize = kwargs.get("figsize", (9, 9))
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


def plot_weight_hist(filename, datadescid, log=False, **kwargs):

    ms.open(filename)
    ms.selectinit(datadescid=datadescid)
    q = ms.getdata(["weight"])
    ms.selectinit(reset=True)
    ms.close()


    weight_XX, weight_YY = q["weight"]
    scatter_XX, scatter_YY = process.weight_to_sigma(q["weight"])

    figsize = kwargs.get("figsize", (9, 9))
    bins = kwargs.get("bins", 20)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=figsize)

    ax[0, 0].hist(weight_XX, bins=bins, log=log)
    ax[0, 0].set_xlabel(r"$w_\mathrm{XX}\,[1/\mathrm{Jy}^2]$")
    ax[0, 1].hist(weight_YY, bins=bins, log=log)
    ax[0, 1].set_xlabel(r"$w_\mathrm{YY}\,[1/\mathrm{Jy}^2]$")

    ax[1, 0].hist(scatter_XX, bins=bins, log=log)
    ax[1, 0].set_xlabel(r"$\sigma_\mathrm{XX}\,[\mathrm{Jy}]$")
    ax[1, 1].hist(scatter_YY, bins=bins, log=log)
    ax[1, 1].set_xlabel(r"$\sigma_\mathrm{YY}\,[\mathrm{Jy}]$")
    fig.subplots_adjust(hspace=0.25)
    fig.suptitle("DATA_DESC_ID: {:}".format(datadescid))

    return fig
