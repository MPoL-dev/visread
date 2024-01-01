import numpy as np
from astropy.constants import c


def doppler_shift(freq, v):
    """
    Calculate the relativistic doppler shift. Negative velocities mean that the frequencies will be blueshifted.

    Args:
        freq (float or np.array): the frequencies in units of Hz.
        v (float or np.array): the velocities in units of m/s. If either `freq` or `v` are arrays, they must have broadcast-able shapes.

    Returns:
        shifted frequencies
    """

    beta = v / c.value

    return np.sqrt((1 - beta) / (1 + beta)) * freq  # Hz


def gaussian(x, sigma=1):
    r"""
    Evaluate a reference Gaussian as a function of :math:`x`

    Args:
        x (float): location to evaluate Gaussian
        sigma (float): standard deviation of Gaussion (default 1)

    The Gaussian is defined as

    .. math::

        f(x) = \frac{1}{\sigma \sqrt{2 \pi}} \exp \left ( -\frac{x^2}{2 \sigma^2}\right )

    Returns:
        Gaussian function evaluated at :math:`x`
    """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / sigma) ** 2)
