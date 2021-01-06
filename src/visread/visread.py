import numpy as np
import casatools

# initialize the relevant CASA tools
tb = casatools.table()
ms = casatools.ms()


class Cube:
    r"""
    Storage container for a set of visibilities.

    Args:
        frequencies (1d numpy array): shape (nchan,) numpy vector of frequencies corresponding to each channel in units of [GHz]. Should be in *decreasing* order such that the visibilities are ordered blueshifted to redshifted with increasing channel index.
        uu (2d numpy array): shape (nchan, nvis) numpy array of east-west spatial frequencies (units of [:math:`\mathrm{k}\lambda`])
        vv (2d numpy array): shape (nchan, nvis) numpy array of north-south spatial frequencies (units of [:math:`\mathrm{k}\lambda`])
        weights (2d numpy array): thermal weights of visibilities (units of [:math:`1/\mathrm{Jy}^2`])
        data_re (2d numpy array): real component of visibility data (units [:math:`\mathrm{Jy}`])
        data_im (2d numpy array): imaginary component of visibility data (units [:math:`\mathrm{Jy}`])
        CASA_convention (boolean): do the baseline conventions follow the `CASA convention <https://casa.nrao.edu/casadocs/casa-5.6.0/memo-series/casa-memos/casa_memo2_coordconvention_rau.pdf>`_ (``CASA_convention==True``; ) or the standard radio astronomy convention (``CASA_convention==False``, i.e., `Thompson, Moran, and Swenson <https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract>`_ Fig 3.2)?
    """

    def __init__(
        self,
        frequencies,
        uu,
        vv,
        weights,
        data_re,
        data_im,
        CASA_convention=True,
        **kwargs
    ):

        assert (
            frequencies.ndim == 1
        ), "frequencies should be a 1D numpy array in decreasing order (blueshifted to redshifted)"
        nchan = len(frequencies)

        assert np.all(
            np.diff(frequencies) < 0.0
        ), "frequencies need to be stored in decreasing order"

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

        self.CASA_convention = CASA_convention

        # parse kwargs for metadata

    def conjugate(self, CASA_convention):
        if self.CASA_convention == CASA_convention:
            print(
                "Visibilities are already set with CASA_convention == {:}".format(
                    CASA_convention
                )
            )
        else:
            print(
                "Swapping CASA_convention from {:} to {:}".format(
                    self.CASA_convention, CASA_convention
                )
            )
            self.data_im = -1.0 * self.data_im


def read(filename, average_polarizations=True):
    """
    Attempt to read the visibilities and some metadata directly from a CASA measurement set.
    """

    pass
