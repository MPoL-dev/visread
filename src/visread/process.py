import numpy as np
from astropy.constants import c

def weight_to_sigma(weight):
    r"""
    Convert a weight (:math:`w`) to an uncertainty (:math:`\sigma`) using

    .. math::

        \sigma = \sqrt{1/w}

    Args:
        weight (float or np.array): statistical weight value

    Returns:
        sigma (float or np.array): the corresponding uncertainty
    """

    return np.sqrt(1 / weight)


def broadcast_weights(weight, data_shape, chan_axis=1):
    r"""
    Broadcast a vector of non-channelized weights to match the shape of the visibility data that is channelized (e.g., for spectral line applications).

    Args:
        weight (np.array): the weights
        data_shape (tuple): the shape of the data
        chan_axis (int): which axis represents the number of channels?

    """

    nchan = data_shape[chan_axis]

    broadcast = np.ones((2, nchan, 1))
    return weight[:, np.newaxis, :] * broadcast


def rescale_weights(weight, sigma_rescale):
    r"""
    Rescale all weights by a common factor. It would be as if :math:`\sigma` were rescaled by this factor.
    
    .. math::

        w_\mathrm{new} = w_\mathrm{old} / \sigma_\mathrm{rescale}^2

    Args:
        weight (float or np.array): the weights
        sigma_rescale (float): the factor by which to rescale the weight 

    Returns:
        (float or np.array) the rescaled weights
    """
    return weight / (sigma_rescale**2)


def sum_weight_polarization(weight, polarization_axis=0):
    """
    Average the weights over the polarization axis.

    Args:
        weights (np.array): weight array. Could be shape `(2, nchan, nvis)` or just `(2, nvis)`, dependending on whether it has been broadcasted already. 
        polarization_axis (int): the polarization axis, typically 0.

    Returns:
        (np.array): weight array summed over the polarization axis. Could be shape `(nchan, nvis)` or just `(nvis)` depending on whether it was broadcasted across channels.
    """

    return np.sum(weight, axis=polarization_axis)


def convert_baselines(baselines, freq):
    r"""
    Convert baselines in meters to kilolambda.

    Args:
        baselines (float or np.array): baselines in [m].
        freq (float or np.array): frequencies in [Hz]. If either ``baselines`` or ``freq`` are numpy arrays, their shapes must be broadcast-able.

    Returns:
        (1D array nvis): baselines in [klambda]
    """
    # calculate wavelengths in meters
    wavelengths = c.value / freq  # m

    # calculate baselines in klambda
    return 1e-3 * baselines / wavelengths  # [klambda]


def broadcast_baselines(u, v, chan_freq):
    r"""
    Convert baselines to kilolambda and broadcast

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
    wavelengths = c.value / chan_freq[:, np.newaxis]  # m

    # calculate baselines in klambda
    uu = 1e-3 * uu / wavelengths  # [klambda]
    vv = 1e-3 * vv / wavelengths  # [klambda]

    return (uu, vv)


def average_data_polarizations(data, weight, flag=None, polarization_axis=0):
    """
    Perform a weighted average of the data over the polarization axis and propagate flags, if necessary.

    Args:
        data (npol, nchan, nvis): complex data array. Could either be real data or model_data.
        weight (npol, nchan, nvis): weight array
        flag (npol, nchan, nvis): bool flag array
        polarization_axis (int): index of the polarization axis, typically 0.

    Returns:
        data, flag, weight averaged over the polarization axis each (nchan, nvis)
    """
    assert data.shape[0] == 2, "Not recognized as a dual-polarization dataset"

    data = np.sum(data * weight, axis=0) / np.sum(weight, axis=0)

    if model_data is not None:
        model_data = np.sum(model_data * weight, axis=0) / np.sum(weight, axis=0)

    flag = np.any(flag, axis=0)
    weight = np.sum(weight, axis=0)

    if model_data is not None:
        return data, flag, weight, model_data
    else:
        return data, flag, weight


def get_crosscorrelation_mask(ant1, ant2):

    # index to cross-correlations
    xc = np.where(ant1 != ant2)[0]

    return xc


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


def get_processed_visibilities(
    filename, datadescid, sigma_rescale=1.0, model_data=False
):
    r"""
    Get all of the visibilities from a specific datadescid. Average polarizations.

    Args:
        filename (str): path to measurementset to process
        datadescid (int): a specific datadescid to process
        sigma_rescale (float): by what factor should the sigmas be rescaled (applied to weights via ``rescale_weights``)
        model_data (bool): include the model_data column?

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
    data, flag, weight, model_data = average_polarizations(
        data, flag, weight, model_data
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
