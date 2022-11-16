import numpy as np
import casatools
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# initialize the relevant CASA tools
tb = casatools.table()
ms = casatools.ms()
msmd = casatools.msmetadata()

c_ms = 2.99792458e8  # [m s^-1]

def doppler_shift(freq, v):
    """
    Calculate the relativistic doppler shift. Negative velocities mean blueshift.

    Args:
        freq (Hz):
        v (m/s):

    Returns:
        shifted frequencies
    """
    
    beta = v / c_ms

    return np.sqrt((1 - beta)/(1 + beta)) * freq # Hz


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


