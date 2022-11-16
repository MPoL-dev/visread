import numpy as np
import casatools
from scipy.optimize import minimize

# initialize the relevant CASA tools
tb = casatools.table()
ms = casatools.ms()
msmd = casatools.msmetadata()


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
