import numpy as np
from scipy.optimize import minimize

from . import utils, process

def get_scatter(
    data, weight, flag, model=None, sigma_rescale=1.0, apply_flags=True, residual=True
):
    if residual:
        residuals = data - model
    else:
        residuals = data

    # calculate sigma from weight
    sigma = process.weight_to_sigma(weight)
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


def calculate_rescale_factor(s, method="Nelder-Mead", bins=40):
    """
    Calculate the multiplicative factor needed to scale :math:`\sigma` such that the scatter in the residuals matches that expected from a Gaussian.

    Args:
        s (np.array): an array of residuals normalized to their :math:`\sigma` values. Assumes values are float.
        method (string): string passed to `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ to choose minimization argument.
        bins (int): number of bins to use in the histogram

    Returns:
        float: the multiplicative factor needed to multiply against :math:`\sigma`
    """

    if s.dtype == np.complex128:
        raise RuntimeError("You passed a complex-valued scatter quantity to scatter.calculate_rescale_factor. This routine only works on float, so if you have a complex data type, pass .real and .imag components on separate invocations.")

    # create a histogram of the scatter values, should approximate a Gaussian distribution
    bin_heights, bin_edges = np.histogram(s, density=True, bins=bins)
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


def get_averaged_scatter(data, model_data, weight, flag=None):
    """
    Calculate the scatter of the residual visibilities, assuming they have already been averaged across polarization.

    Args:
        data (np.array complex): the data visibilities
        model_data (np.array complex): the model visibilities
        weight (np.array real): the statistical weight of the uncertainties
        flag (np.array bool): the flags of the dataset, in original format (``True`` should be flagged).

    Returns:
        np.array: the scatter of the residual visibilities in units of :math:`\sigma`
    """

    residuals = data - model_data
    sigma = process.weight_to_sigma(weight)

    scatter = residuals / sigma

    if flag is not None:
        return scatter[~flag]
    else:
        return scatter
