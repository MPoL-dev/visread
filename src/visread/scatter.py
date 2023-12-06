import numpy as np
import casatools
from scipy.optimize import minimize

from . import utils, process

# initialize the relevant CASA tools
msmd = casatools.msmetadata()
ms = casatools.ms()

def get_scatter_datadescid(filename, datadescid, sigma_rescale=1.0, apply_flags=True, residual=True, datacolumn="corrected_data"):
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
        residual (bool): if True, subtract MODEL_DATA column (from a tclean model, most likely) to plot scatter of residual visibilities.
        datacolumn (string): which datacolumn to use (i.e., 'corrected_data' or 'data').

    Returns:
        scatter_XX, scatter_YY: a 2-tuple of numpy arrays containing the scatter in each polarization.
        If ``apply_flags==True``, each array will be 1-dimensional. If ``apply_flags==False``, each array
        will retain its original shape, including channelization (e.g., shape ``nchan,nvis``).

    """


    ms.open(filename)
    ms.selectinit(datadescid=datadescid)
    keys = ["weight", "flag", datacolumn]
    if residual:
        keys += ["model_data"]
    q = ms.getdata(keys)
    ms.selectinit(reset=True)
    ms.close()

    if residual:
        assert (
            len(q["model_data"]) > 0
        ), "MODEL_DATA column empty, retry tclean with savemodel='modelcolumn'"

        # subtract model from data
        residuals = q[datacolumn] - q["model_data"]

    else:
        # assume the S/N of each individual visibility is negligible. 
        residuals = q[datacolumn]

    # calculate sigma from weight
    sigma = process.weight_to_sigma(q["weight"])
    sigma *= sigma_rescale

    # divide by weight, augmented for channel dim
    scatter = residuals / sigma[:, np.newaxis, :]

    # separate polarizations
    scatter_XX, scatter_YY = scatter
    flag_XX, flag_YY = q["flag"]

    if apply_flags:
        # flatten across channels
        scatter_XX = scatter_XX[~flag_XX]
        scatter_YY = scatter_YY[~flag_YY]

    return scatter_XX, scatter_YY

def calculate_rescale_factor(scatter, method="Nelder-Mead", bins=40):
    """
    Calculate the multiplicative factor needed to scale :math:`\sigma` such that the scatter in the residuals matches that expected from a Gaussian.

    Args:
        scatter (np.array): an array of residuals normalized to their :math:`\sigma` values.
        method (string): string passed to `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ to choose minimization argument.
        bins (int): number of bins to use in the histogram

    Returns:
        float: the multiplicative factor needed to multiply against :math:`\sigma`
    """

    # create a histogram of the scatter values, should approximate a Gaussian distribution
    bin_heights, bin_edges = np.histogram(scatter, density=True, bins=bins)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    # find the sigma_rescale which minimizes the mean squared error
    # between the bin_heights and the expectations from the
    # reference Gaussian
    loss = lambda x: np.sum((bin_heights - utils.gaussian(bin_centers, sigma=x)) ** 2)

    res = minimize(loss, 1.0, method=method)

    if res.success:
        return res.x[0]
    else:
        print(res)
        return False


def get_sigma_rescale_datadescid(filename, datadescid, datacolumn="corrected_data"):
    """
    For a given datadescid, calculate the residual scatter in each of the XX and YY polarization visibilities, then calculate the sigma rescale factor for each of the real and imaginary values of the polarizations. Return the average of all four quantities as the final sigma rescale factor for that datadescid.

    Args:
        filename (string): path to measurement set
        datadescid (int): the spectral window in the measurement set
        
    Returns:
        float: the multiplicative factor by which to scale :math:`\sigma`
    """
    scatter_XX, scatter_YY = get_scatter_datadescid(
        filename, datadescid, apply_flags=True, datacolumn=datacolumn)

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

def get_averaged_scatter(data, model_data, weight, flag=None):
    """
    Calculate the scatter of the residual visibilities, assuming they have already been averaged across polarization.

    Args:
        data (np.array complex): the data visibilities
        model_data (np.array complex): the model visibilities
        weight (np.array real): the statistical weight of the uncertainties
        flag (np.array bool): the flags of the dataset, in original format (``True`` should be flagged).

    Returns:
        np.arary: the scatter of the residual visibilities in units of :math:`\sigma`
    """

    residuals = data - model_data
    sigma = process.weight_to_sigma(weight)

    scatter = residuals / sigma

    if flag is not None:
        return scatter[~flag]
    else:
        return scatter
