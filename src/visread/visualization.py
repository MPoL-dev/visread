import numpy as np
import matplotlib.pyplot as plt
from . import utils, scatter, process


def plot_baselines(u, v, title=None):
    """
    Make a plot of the baselines.

    Args:
        u: baseline in meters
        v: baseline in meters.

    Returns:
        matplotlib.fig

    """
    fig, ax = plt.subplots(nrows=1)
    ax.scatter(u * 1e-3, v * 1e-3, s=0.5)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$u$ [km]")
    ax.set_ylabel(r"$v$ [km]")

    return fig


def scatter_hist(scatter_XX, scatter_YY, log=False, **kwargs):
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
        a.plot(xs, utils.gaussian(xs))

    fig.subplots_adjust(hspace=0.25, top=0.95)

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

def plot_weight_hist(weight_XX, weight_YY, log=False, title=None, **kwargs):
    sigma_XX = process.weight_to_sigma(weight_XX)
    sigma_YY = process.weight_to_sigma(weight_YY)

    figsize = kwargs.get("figsize", (9, 9))
    bins = kwargs.get("bins", 20)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=figsize)

    ax[0, 0].hist(weight_XX, bins=bins, log=log)
    ax[0, 0].set_xlabel(r"$w_\mathrm{XX}\,[1/\mathrm{Jy}^2]$")
    ax[0, 1].hist(weight_YY, bins=bins, log=log)
    ax[0, 1].set_xlabel(r"$w_\mathrm{YY}\,[1/\mathrm{Jy}^2]$")

    ax[1, 0].hist(sigma_XX, bins=bins, log=log)
    ax[1, 0].set_xlabel(r"$\sigma_\mathrm{XX}\,[\mathrm{Jy}]$")
    ax[1, 1].hist(sigma_YY, bins=bins, log=log)
    ax[1, 1].set_xlabel(r"$\sigma_\mathrm{YY}\,[\mathrm{Jy}]$")
    fig.subplots_adjust(hspace=0.25)
    if title is not None:
        fig.suptitle(title)

    return fig
