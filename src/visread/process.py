import numpy as np
from astropy.constants import c


def convert_baselines(baselines, freq):
    r"""
    Convert baselines in meters to kilolambda. Assumes that baselines and freq will broadcast under division.

    Args:
        baselines (float or np.array): baselines in [m].
        freq (float or np.array): frequencies in [Hz]. If either ``baselines`` or ``freq`` are numpy arrays, their shapes must be broadcast-able.

    Returns:
        (1D array nvis): baselines in [lambda]
    """
    # calculate wavelengths in meters
    wavelengths = c.value / freq  # m

    # calculate baselines in lambda
    return baselines / wavelengths  # [lambda]


def broadcast_and_convert_baselines(u, v, chan_freq):
    r"""
    Convert baselines to lambda and broadcast to match shape of channel frequencies.

    Args:
        u (1D array nvis): baseline [m]
        v (1D array nvis): baseline [m]
        chan_freq (1D array nchan): frequencies [Hz]

    Returns:
        (u, v) each of which are (nchan, nvis) arrays of baselines in [lambda]
    """

    nchan = len(chan_freq)

    # broadcast to the same shape as the data
    # stub to broadcast u, v to all channels
    broadcast = np.ones((nchan, 1))
    uu = u * broadcast
    vv = v * broadcast

    # calculate wavelengths in meters
    wavelengths = c.value / chan_freq[:, np.newaxis]  # m

    # calculate baselines in lambda
    uu = uu / wavelengths  # [lambda]
    vv = vv / wavelengths  # [lambda]

    return (uu, vv)


def broadcast_weights(weight, data_shape, chan_axis=0):
    r"""
    Broadcast a vector of non-channelized weights to match the shape of the visibility data that is channelized (e.g., for spectral line applications) but already averaged over polarizations.

    Args:
        weight (np.array): the weights
        data_shape (tuple): the shape of the data
        chan_axis (int): the axis which represents the number of channels in the data array, typically 0 for visibility data that has already been averaged over polarizations.

    Returns:
        np.array (float) array of weights the same shape as the data
    """

    nchan = data_shape[chan_axis]

    broadcast = np.ones((nchan, 1))
    return weight[np.newaxis, :] * broadcast


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


def average_data_polarization(data, weight, polarization_axis=0):
    """
    Perform a weighted average of the data over the polarization axis.

    Args:
        data (npol, nchan, nvis): complex data array. Could either be real data or model_data.
        weight (npol, nvis): weight array matching data array (before broadcast)
        polarization_axis (int): index of the polarization axis, typically 0.

    Returns:
        data averaged over the polarization axis.
    """
    assert (
        data.shape[polarization_axis] == 2
    ), "Not recognized as a dual-polarization dataset"

    # we need to check whether weight is the same shape as the data, because sometimes the data is
    # channelized and the weights are not
    # e.g., data would have shape (npol, nchan, nvis)
    # while weight would have shape (npol, nvis)

    # normalization after averaging over the polarization axis
    norm = average_weight_polarization(weight, polarization_axis=polarization_axis)

    if len(data.shape) == len(weight.shape):
        return np.sum(data * weight, axis=polarization_axis) / norm
    elif (len(data.shape) == 3) and (len(weight.shape) == 2):
        return np.sum(data * weight[:, np.newaxis, :], axis=polarization_axis) / norm
    else:
        raise RuntimeError(
            "I don't know what to do with provided data and weight arrays with shapes {:} and {:}, respectively".format(
                data.shape, weight.shape
            )
        )


def average_weight_polarization(weight, polarization_axis=0):
    """
    Average the weights over the polarization axis.

    Args:
        weight (np.array): weight array. Could be shape `(2, nchan, nvis)` or just `(2, nvis)`, dependending on whether it has been broadcasted already.
        polarization_axis (int): the polarization axis, typically 0.

    Returns:
        (np.array): weight array summed over the polarization axis. Could be shape `(nchan, nvis)` or just `(nvis)` depending on whether it was broadcasted across channels.
    """

    return np.sum(weight, axis=polarization_axis)


def average_flag_polarization(flag, polarization_axis=0):
    """
    Collapse the flags across the polarization axis, taking the approach that if either polarization is flagged, the averaged product shoud be flagged too.

    Args:
        flag (np.array bool): flag array. Could be multidimensional, e.g. `(2, nchan, nvis)` or just `(2, nvis)`.
        polarization_axis (int): the polarization axis, typically 0.

    Returns:
        (np.array bool): flag array collapsed across the polarization axis. Could be shape `(nchan, nvis)` or just `(nvis)` depending on whether it was broadcasted across channels.

    """
    return np.any(flag, axis=polarization_axis)


def contains_autocorrelations(ant1, ant2):
    """
    Test whether the list of antennas contain any autocorrelations.

    Args:
        ant1 (np.array int): antenna 1
        ant2 (np.array int): antenna 2

    Returns:
        boolean: True if list contains autocorrelation pairs.
    """
    autocorrelation_mask = ant1 == ant2
    return np.sum(autocorrelation_mask) > 0


def get_crosscorrelation_indexes(ant1, ant2):
    # index to cross-correlations
    xc = np.where(ant1 != ant2)[0]

    return xc


def isdecreasing(chan_freq):
    """
    Return true if channels are stored in decreasing frequency order, i.e., blueshifted to redshifted.

    Args:
        chan_freq (1D numpy array): the channel frequencies in Hz

    Returns:
        boolean : True if channels are stored in decreasing frequency order (preferred order).
    """
    # check to make sure we're in blushifted - redshifted order, otherwise reverse channel order
    nchan = len(chan_freq)
    if nchan == 1:
        return True

    diff = np.diff(chan_freq)

    if np.all(diff < 0):
        return True  # strictly decreasing
    elif np.all(diff > 0):
        return False  # strictly increasing
    else:
        raise RuntimeError(
            "chan_freq array is neither strictly decreasing nor strictly increasing, investigate what went wrong."
        )
