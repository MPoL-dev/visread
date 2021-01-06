import numpy as np
import casatools


class Cube:
    """

    frequencies: 1d numpy array
    uu: 2d numpy array
    """

    def __init__(self, frequencies, uu, vv, weights, data_re, data_im, **kwargs):

        assert frequencies.ndim == 1, "frequencies should be a 1D numpy array"
        nchan = len(frequencies)

        assert uu.ndim == 2, "uu should be a 2D numpy array"
        shape = uu.shape

        for a in [vv, weights, data_re, data_im]:
            assert a.shape == shape, "All dataset inputs must be the same 2D shape."

        assert np.all(
            weights > 0.0
        ), "Not all thermal weights are positive, check inputs."

        assert data_re.dtype == np.float64, "data_re should be type np.float64"
        assert data_im.dtype == np.float64, "data_im should be type np.float64"

        self.nchan = nchan
        self.frequencies = frequencies
        self.uu = uu
        self.vv = vv
        self.weights = weights
        self.data_re = data_re
        self.data_im = data_im

        # parse kwargs


def read(filename):
    """
    Attempt to read the visibilities and some metadata directly from a CASA measurement set.
    """

    # check number of spectral windows

    # check flagged visibilities

    pass
