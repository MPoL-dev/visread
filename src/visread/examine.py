import numpy as np
import casatools
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# initialize the relevant CASA tools
tb = casatools.table()
ms = casatools.ms()
msmd = casatools.msmetadata()

c_ms = 2.99792458e8  # [m s^-1]


def weight_to_sigma(weight):
    r"""
    Convert a weight (:math:`w`) to an uncertainty (:math:`\sigma`) using

    .. math::

        \sigma = \sqrt{1/w}

    Args:
        weight (float): statistical weight value

    Returns:
        sigma (float): the corresponding uncertainty
    """

    return np.sqrt(1 / weight)


def gaussian(x, sigma=1):
    r"""
    Evaluate a reference Gaussian as a function of :math:`x`

    Args:
        x (float): location to evaluate Gaussian
        sigma (float): standard deviation of Gaussion (default 1)

    The Gaussian is defined as

    .. math::

        f(x) = \frac{1}{\sqrt{2 \pi}} \exp \left ( -\frac{x^2}{2}\right )

    Returns:
        Gaussian function evaluated at :math:`x`
    """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / sigma) ** 2)


def get_colnames(filename):
    """
    Examine the column names of a measurement set.

    Args:
        filename (string): measurement set filename

    Returns:
        list of column names
    """

    tb.open(filename)
    colnames = tb.colnames()
    tb.close()

    return colnames


def get_unique_datadescid(filename):
    """
    Get the list of unique ``DATA_DESC_ID`` present in a measurement set.

    Args:
        filename (string): measurement set filename

    Returns:
        Array of ``DATA_DESC_ID`` values contained in the measurement set.
    """

    tb.open(filename)
    spw_id = tb.getcol("DATA_DESC_ID")  # array of int with shape [npol, nchan, nvis]
    tb.close()

    return np.unique(spw_id)


def query_datadescid(
    filename,
    datadescid,
    colnames=["DATA", "MODEL_DATA", "WEIGHT", "UVW", "ANTENNA1", "ANTENNA2", "FLAG"],
):
    """
    Use the ``ms`` casatool to query the measurement set for quantities pertaining to a specific ``DATA_DESC_ID``.

    Args:
        filename (string): measurement set filename
        datadescid (int): the DATA_DESC_ID to be queried
        colnames (list of strings): list of column names to query

    Returns:
        a dictionary with the column values, hashed by lowercase column name
    """
    # https://casa.nrao.edu/casadocs-devel/stable/global-tool-list/tool_ms/methods
    ms.open(filename)
    # select the key
    ms.selectinit(datadescid=datadescid)
    query = ms.getdata(colnames)
    ms.selectinit(reset=True)
    ms.close()

    return query


def get_channels(filename, datadescid):
    # https://casa.nrao.edu/casadocs-devel/stable/global-tool-list/tool_msmetadata/methods
    msmd.open(filename)
    chan_freq = msmd.chanfreqs(datadescid)
    msmd.done()

    return chan_freq


def get_channel_sorted_data(filename, datadescid):
    # get the channels
    chan_freq = get_channels(filename, datadescid)
    nchan = len(chan_freq)

    # get the data and flags
    query = query_datadescid(filename, datadescid)
    data = query["data"]
    model_data = query["model_data"]
    flag = query["flag"]

    # check to make sure we're in blushifted - redshifted order, otherwise reverse channel order
    if (nchan > 1) and (chan_freq[1] > chan_freq[0]):
        # reverse channels
        chan_freq = chan_freq[::-1]
        data = data[:, ::-1, :]
        model_data = model_data[:, ::-1, :]
        flag = flag[:, ::-1, :]

    return chan_freq, data, model_data, flag


def broadcast_baselines(u, v, chan_freq):
    r"""
    Convert baselines to kilolambda

    Args:
        u (1D array nvis): baseline [m]
        v (1D array nvis): baseline [m]
        chan_freq (1D array nchan): frequencies [Hz]

    Returns:
        (u, v) each of which are (nchan, nvis) arrays of baselines in [klambda]
    """

    nchan = len(chan_freq)

    # broadcast to the same shape as the data
    # stub to broadcast u, v to all channels
    broadcast = np.ones((nchan, 1))
    uu = u * broadcast
    vv = v * broadcast

    # calculate wavelengths in meters
    wavelengths = c_ms / chan_freq[:, np.newaxis]  # m

    # calculate baselines in klambda
    uu = 1e-3 * uu / wavelengths  # [klambda]
    vv = 1e-3 * vv / wavelengths  # [klambda]

    return (uu, vv)


def broadcast_weights(weight, nchan):

    # weight is shape (npol, nvis)

    # we want to make it shape (npol, nchan, nvis)

    broadcast = np.ones((2, nchan, 1))
    return weight[:, np.newaxis, :] * broadcast


def average_polarizations(data, model_data, flag, weight):
    """
    Average over the polarization axis.

    Args:
        data (2, nchan, nvis): complex data array
        flag (2, nchan, nvis): bool flag array
        weight (2, nchan, nvis): assume it's already been broadcasted

    Returns:
        data, flag, weight averaged over the polarization axis each (nchan, nvis)
    """
    assert data.shape[0] == 2, "Not recognized as a dual-polarization dataset"

    data = np.sum(data * weight, axis=0) / np.sum(weight, axis=0)
    model_data = np.sum(model_data * weight, axis=0) / np.sum(weight, axis=0)
    flag = np.any(flag, axis=0)
    weight = np.sum(weight, axis=0)

    return data, model_data, flag, weight


def get_crosscorrelation_mask(ant1, ant2):

    # index to cross-correlations
    xc = np.where(ant1 != ant2)[0]

    return xc


def rescale_weights(weight, sigma_rescale):
    return weight / (sigma_rescale ** 2)


def get_processed_visibilities(filename, datadescid, sigma_rescale=1.0):
    r"""
    Get all of the visibilities from a specific datadescid. Average polarizations.

    Returns:
        dictionary with keys "frequencies", "uu", "data", "flag", "weight"


    """
    # get sorted channels, data, and flags
    chan_freq, data, model_data, flag = get_channel_sorted_data(filename, datadescid)
    nchan = len(chan_freq)

    # get baselines, weights, and antennas
    query = query_datadescid(filename, datadescid)

    # broadcast baselines
    uu, vv, ww = query["uvw"]  # [m]
    uu, vv = broadcast_baselines(uu, vv, chan_freq)

    # broadcast and rescale weights
    weight = query["weight"]
    weight = broadcast_weights(weight, nchan)
    weight = rescale_weights(weight, sigma_rescale)

    # average polarizations
    data, model_data, flag, weight = average_polarizations(
        data, model_data, flag, weight
    )

    # calculate the cross correlation mask
    ant1 = query["antenna1"]
    ant2 = query["antenna2"]
    xc = get_crosscorrelation_mask(ant1, ant2)

    # apply the xc mask across channels
    # drop autocorrelation channels
    uu = uu[:, xc]
    vv = vv[:, xc]
    data = data[:, xc]
    model_data = model_data[:, xc]
    flag = flag[:, xc]
    weight = weight[:, xc]

    # take the complex conjugate
    data = np.conj(data)
    model_data = np.conj(model_data)

    return {
        "frequencies": chan_freq,
        "uu": uu,
        "vv": vv,
        "data": data,
        "model_data": model_data,
        "flag": flag,
        "weight": weight,
    }


def get_scatter_datadescid(filename, datadescid, sigma_rescale=1.0, apply_flags=True):
    r"""
    Calculate the residuals for each polarization (XX, YY) in units of :math:`\sigma`, where

    .. math::

        \sigma = \mathrm{sigma\_rescale} \times \sigma_0

    and :math:`\sigma_0 = \sqrt{1/w}`. The scatter is defined as

    .. math::

        \mathrm{scatter} = \frac{\mathrm{DATA} - \mathrm{MODEL\_DATA}}{\sigma}

    Args:
        filename (string): measurement set filename
        datadescid (int): the DATA_DESC_ID to be queried
        sigma_rescale (int):  multiply the uncertainties by this factor
        apply_flags (bool): calculate the scatter *after* the flags have been applied

    Returns:
        scatter_XX, scatter_YY: a 2-tuple of numpy arrays containing the scatter in each polarization.
        If ``apply_flags==True``, each array will be 1-dimensional. If ``apply_flags==False``, each array
        will retain its original shape, including channelization (e.g., shape ``nchan,nvis``).

    """

    query = query_datadescid(filename, datadescid)
    data, model_data, weight, flag = (
        query["data"],
        query["model_data"],
        query["weight"],
        query["flag"],
    )

    # TODO: assert model_data not None?
    assert (
        len(model_data) > 0
    ), "MODEL_DATA column empty, retry tclean with savemodel='modelcolumn'"

    # subtract model from data
    residuals = data - model_data

    # calculate sigma from weight
    sigma = weight_to_sigma(weight)
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

        scatter_XX, scatter_YY = get_scatter_datadescid(
            filename, datadescid, sigma_rescale, apply_flags=False
        )

        scatter_XX = scatter_XX[chan_slice]
        scatter_YY = scatter_YY[chan_slice]

    else:
        scatter_XX, scatter_YY = get_scatter_datadescid(
            filename, datadescid, sigma_rescale, apply_flags=apply_flags
        )

    scatter_XX = scatter_XX.flatten()
    scatter_YY = scatter_YY.flatten()

    fig = _scatter_hist(scatter_XX, scatter_YY, log=log)
    fig.suptitle("DATA_DESC_ID: {:}".format(datadescid))

    return fig


def get_averaged_scatter(d):
    """
    Args:
        d : dictionary with keys
    """

    residuals = d["data"] - d["model_data"]
    sigma = weight_to_sigma(d["weight"])

    scatter = residuals / sigma

    # apply flags
    flag = d["flag"]

    return scatter[~flag]


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
        a.plot(xs, gaussian(xs))

    fig.subplots_adjust(hspace=0.25, top=0.95)

    return fig


def plot_baselines(filename, datadescid):

    query = query_datadescid(filename, datadescid)
    u, v, w = query["uvw"] * 1e-3  # [km]

    fig, ax = plt.subplots(nrows=1)
    ax.scatter(u, v, s=0.5)
    ax.set_title("DATA_DESC_ID: {:}".format(datadescid))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$u$ [km]")
    ax.set_ylabel(r"$v$ [km]")

    return fig


def plot_weight_hist(filename, datadescid, log=False, **kwargs):

    query = query_datadescid(filename, datadescid)
    weight = query["weight"]

    weight_XX, weight_YY = weight
    scatter_XX, scatter_YY = weight_to_sigma(weight)

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


def calculate_rescale_factor(scatter, **kwargs):
    bins = kwargs.get("bins", 40)
    bin_heights, bin_edges = np.histogram(scatter, density=True, bins=bins)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    # find the sigma_rescale which minimizes the mean squared error
    # between the bin_heights and the expectations from the
    # reference Gaussian

    loss = lambda x: np.sum((bin_heights - gaussian(bin_centers, sigma=x)) ** 2)

    res = minimize(loss, 1.0)

    if res.success:
        return res.x[0]
    else:
        print(res)
        return False


def get_sigma_rescale_datadescid(filename, datadescid, **kwargs):
    scatter_XX, scatter_YY = get_scatter_datadescid(
        filename, datadescid, apply_flags=True, **kwargs
    )

    vals = np.array(
        [
            calculate_rescale_factor(scatter)
            for scatter in [
                scatter_XX.real,
                scatter_XX.imag,
                scatter_YY.real,
                scatter_YY.imag,
            ]
        ]
    )

    return np.average(vals)


# * broadcast baselines (to klambda)
# * average polarizations
