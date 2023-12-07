import numpy as np
from . import process

try:
    import casatools

    # initialize the relevant CASA tools
    msmd = casatools.msmetadata()
    ms = casatools.ms()
except ModuleNotFoundError as e:
    print(
        "casatools module not found on system. If your system configuration is compatible, you can try installing these optional dependencies with `pip install 'visread[casa]'`. More information on Modular CASA can be found https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Modular-Packages "
    )
    raise e


def get_channel_sorted_data(
    filename, datadescid, incl_model_data=True, datacolumn="corrected_data"
):
    """
    Acquire and sort the channel frequencies, data, flags, and model_data columns.

    Args:
        filename (string): the measurement set to query
        datadescid (int): the spw id to query
        incl_model_data (boolean): if ``True``, return the ``model_data`` column as well
        datacolumn (string): "corrected_data" by default

    Returns:
        tuple: chan_freq, data, flag, model_data
    """

    # get the channel frequencies
    msmd.open(filename)
    chan_freq = msmd.chanfreqs(datadescid)
    msmd.done()

    # get the data and flags
    ms.open(filename)
    ms.selectinit(datadescid=datadescid)
    keys = ["flag", datacolumn]
    if incl_model_data:
        keys += ["model_data"]
        q = ms.getdata(keys)
        model_data = q["model_data"]
    else:
        q = ms.getdata(keys)
    ms.selectinit(reset=True)
    ms.close()

    data = q[datacolumn]
    flag = q["flag"]

    # check to make sure we're in blushifted - redshifted order, otherwise reverse channel order
    if (len(chan_freq) > 1) and (chan_freq[1] > chan_freq[0]):
        # reverse channels
        chan_freq = np.flip(chan_freq)
        data = np.flip(data, axis=1)
        flag = np.flip(flag, axis=1)

        if incl_model_data:
            model_data = np.flip(model_data, axis=1)

    if incl_model_data:
        return chan_freq, data, flag, model_data
    else:
        return chan_freq, data, flag, None


def get_processed_visibilities(
    filename,
    datadescid,
    sigma_rescale=1.0,
    incl_model_data=None,
    datacolumn="corrected_data",
):
    r"""
    Process all of the visibilities from a specific datadescid. This means

    * (If necessary) reversing the channel dimension such that channel frequency decreases with increasing array index (blueshifted to redshifted)
    * averaging the polarizations together
    * rescaling weights
    * scanning and removing any auto-correlation visibilities

    Args:
        filename (str): path to measurementset to process
        datadescid (int): a specific datadescid to process
        sigma_rescale (float): by what factor should the sigmas be rescaled (applied to weights via ``rescale_weights``)
        incl_model_data (bool): include the model_data column?

    Returns:
        dictionary with keys "frequencies", "uu", "data", "flag", "weight"


    """

    # get sorted channels, data, and flags
    chan_freq, data, flag, model_data = get_channel_sorted_data(
        filename, datadescid, incl_model_data, datacolumn=datacolumn
    )

    # get baselines, weights, and antennas
    ms.open(filename)
    ms.selectinit(datadescid=datadescid)
    q = ms.getdata(["uvw", "weight", "antenna1", "antenna2", "time"])
    ms.selectinit(reset=True)
    ms.close()

    time = q["time"]

    uu, vv, ww = q["uvw"]  # [m]

    # rescale weights
    weight = q["weight"]
    weight = process.rescale_weights(weight, sigma_rescale)

    # average the data across polarization
    data = process.average_data_polarization(data, weight)
    flag = process.average_flag_polarization(flag)

    if incl_model_data:
        model_data = process.average_data_polarization(model_data, weight)

    # finally average weights across polarization
    weight = process.average_weight_polarization(weight)

    # calculate the cross correlation mask
    ant1 = q["antenna1"]
    ant2 = q["antenna2"]

    # make sure the dataset doesn't contain auto-correlations
    assert not process.contains_autocorrelations(
        ant1, ant2
    ), "Dataset contains autocorrelations, exiting."

    # # apply the xc mask across channels
    # # drop autocorrelation channels
    # uu = uu[:, xc]
    # vv = vv[:, xc]
    # data = data[:, xc]
    # model_data = model_data[:, xc]
    # flag = flag[:, xc]
    # weight = weight[:, xc]

    # take the complex conjugate
    data = np.conj(data)
    if incl_model_data:
        model_data = np.conj(model_data)

    return {
        "frequencies": chan_freq,
        "uu": uu,
        "vv": vv,
        "antenna1": ant1,
        "antenna2": ant2,
        "time": time,
        "data": data,
        "model_data": model_data,
        "flag": flag,
        "weight": weight,
    }
