import numpy as np
import casatools

# initialize the relevant CASA tools
tb = casatools.table()
ms = casatools.ms()
c_ms = 2.99792458e8  # [m s^-1]


class Cube:
    r"""
    Storage container for a set of visibilities.

    Args:
        frequencies (1d numpy array): shape (nchan,) numpy vector of frequencies corresponding to each channel in units of [GHz]. Should be in *decreasing* order such that the visibilities are ordered blueshifted to redshifted with increasing channel index.
        uu (2d numpy array): shape (nchan, nvis) numpy array of east-west spatial frequencies (units of [:math:`\mathrm{k}\lambda`])
        vv (2d numpy array): shape (nchan, nvis) numpy array of north-south spatial frequencies (units of [:math:`\mathrm{k}\lambda`])
        weight (2d numpy array): thermal weights of visibilities (units of [:math:`1/\mathrm{Jy}^2`])
        data_re (2d numpy array): real component of visibility data (units [:math:`\mathrm{Jy}`])
        data_im (2d numpy array): imaginary component of visibility data (units [:math:`\mathrm{Jy}`])
        CASA_convention (boolean): do the baseline conventions follow the `CASA convention <https://casa.nrao.edu/casadocs/casa-5.6.0/memo-series/casa-memos/casa_memo2_coordconvention_rau.pdf>`_ (``CASA_convention==True``; ) or the standard radio astronomy convention (``CASA_convention==False``, i.e., `Thompson, Moran, and Swenson <https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract>`_ Fig 3.2)?

    Examples:
        >>> #After initialization, you can access each of these attributes via their names
        >>> myCube = visread.read("myMeasurementSet.ms")
        >>> print(myCube.vv)
    """

    def __init__(
        self,
        frequencies,
        uu,
        vv,
        weight,
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

        for a in [vv, weight, data_re, data_im]:
            assert a.shape == shape, "All dataset inputs must be the same 2D shape."

        assert np.all(
            weight > 0.0
        ), "Not all thermal weights are positive, check inputs."

        assert data_re.dtype == np.float64, "data_re should be type np.float64"
        assert data_im.dtype == np.float64, "data_im should be type np.float64"

        self.nchan = nchan
        self.frequencies = frequencies
        self.uu = uu
        self.vv = vv
        self.weight = weight
        self.data_re = data_re
        self.data_im = data_im

        self.CASA_convention = CASA_convention

        # parse kwargs for metadata

    def swap_convention(self, CASA_convention):
        r"""
        Change the convention in which visibilities are stored.

        Args:
            CASA_convention (bool): If True, store the visibilities in the `CASA convention <https://casa.nrao.edu/casadocs/casa-5.6.0/memo-series/casa-memos/casa_memo2_coordconvention_rau.pdf>`_. If False, store them in the standard radio astronomy convention (``CASA_convention==False``, i.e., `Thompson, Moran, and Swenson <https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract>`_ Fig 3.2).

        Returns: None

        Examples:
            >>> myCube = visread.read("myMeasurementSet.ms")
            >>> # swap to TMS convention
            >>> myCube.swap_convention(CASA_convention=False)
        """
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


def read(filename, datacolumn="CORRECTED_DATA"):
    """
    Attempt to read the visibilities and some metadata directly from a 'minimal' CASA measurement set.

    Args:
        filename (string): the measurement set path (i.e., `*.ms`)
        datacolumn (string): which datacolumn to read from the measurement set. By default, attempts to read ``CORRECTED_DATA`` first, and falls back to ``DATA`` if that doesn't exist. More information is available on the `CASA docs <https://casa.nrao.edu/casadocs-devel/stable/casa-fundamentals/the-measurementset>`_.

    Returns: Instantiated ``visread.Cube`` object
    """

    # before reading the data itself, we're going to check that the data is in a
    # minimal format, so that we don't have to make too many assumptions about
    # how to average it, etc.

    # we do this by opening up many of the "tables" of the measurement set
    # information on these are in the CASA docs at
    # https://casa.nrao.edu/casadocs-devel/stable/casa-fundamentals/the-measurementset
    # and
    # https://casa.nrao.edu/casadocs-devel/stable/casa-fundamentals/measurement-set

    # information on the table too itself is at
    # https://casa.nrao.edu/casadocs-devel/stable/global-tool-list/tool_table/methods

    # check number of spectral windows is 1
    tb.open(filename + "/DATA_DESCRIPTION")
    SPECTRAL_WINDOW_ID = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()
    assert SPECTRAL_WINDOW_ID == [
        0
    ], "Measurement Set contains more than one spectral window, first average or export the one you'd like to a separate Measurement Set using CASA/split, mstransform, and/or cvel2. Inspect with listobs or browsetable."

    tb.open(filename)
    colnames = tb.colnames()
    spw_id = tb.getcol("DATA_DESC_ID")  # array of int with shape [npol, nchan, nvis]
    field_id = tb.getcol("FIELD_ID")  # array of int with shape [npol, nchan, nvis]
    ant1 = tb.getcol("ANTENNA1")  # array of int with shape [nvis]
    ant2 = tb.getcol("ANTENNA2")  # array of int with shape [nvis]
    uvw = tb.getcol("UVW")  # array of float64 with shape [3, nvis]
    weight = tb.getcol("WEIGHT")  # array of float64 with shape [npol, nvis]
    flag = tb.getcol("FLAG")  # array of bool with shape [npol, nchan, nvis]
    if datacolumn == "CORRECTED_DATA" and datacolumn not in colnames:
        print("Couldn't find CORRECTED_DATA in column names, using DATA instead")
        datacolumn = "DATA"
    data = tb.getcol(datacolumn)  # array of complex128 with shape [npol, nchan, nvis]
    tb.close()

    assert np.unique(spw_id) == np.array(
        [0]
    ), "Measurement Set contains more than one spectral window, first average or export the one you'd like to a separate Measurement Set using CASA/split, mstransform, and/or cvel2. Inspect with listobs or browsetable."

    # check targets are 1
    assert np.unique(field_id) == np.array(
        [0]
    ), "Measurement Set contains more than one spectral window, first average or export the one you'd like to a separate Measurement Set using CASA/split, mstransform, and/or cvel2. Inspect with listobs or browsetable."

    # check there are no flagged visibilities
    assert (
        np.sum(flag) == 0
    ), "Measurement Set contains flagged visibilities. First export the unflagged visibilities to a new Measurement Set using CASA/split or mstransform."

    assert (
        len(data.shape) == 3
    ), "DATA column contains something other than three dimensions (npol, nchan, nvis) and I don't know what to do with this in the context of a data cube."

    assert (
        len(uvw.shape) == 2
    ), "UVW baselines contains something other than two dimensions (3, nvis) and I don't know what to do with this."

    assert (
        len(weight.shape) == 2
    ), "WEIGHT contains something other than two dimensions (npol, nvis), I don't know what to do with this. WEIGHTSPECTRUM functionality yet to be implemented."

    # get the channel information
    tb.open(filename + "/SPECTRAL_WINDOW")
    chan_freq = tb.getcol("CHAN_FREQ")
    num_chan = tb.getcol("NUM_CHAN")
    tb.close()

    assert (
        len(num_chan) == 1
    ), "More than one spectral window or field still remains in Measurement Set, difficulty reading NUM_CHAN from SPECTRAL_WINDOW table. Inspect with listobs or browsetable."

    assert (
        chan_freq.shape[1] == 1
    ), "More than one spectral window or field still remains in Measurement Set, difficulty reading CHAN_FREQ from SPECTRAL_WINDOW table. Inspect with listobs or browsetable."

    chan_freq = chan_freq.flatten()  # Hz
    nchan = len(chan_freq)

    # check to make sure we're in blushifted - redshifted order, otherwise reverse channel order
    if (nchan > 1) and (chan_freq[1] > chan_freq[0]):
        # reverse channels
        chan_freq = chan_freq[::-1]
        data = data[:, ::-1, :]

    # keep only the cross-correlation visibilities
    # and throw out the auto-correlation visibilities (i.e., where ant1 == ant2)
    xc = np.where(ant1 != ant2)[0]
    data = data[:, :, xc]
    uvw = uvw[:, xc]
    weight = weight[:, xc]

    assert np.all(
        weight > 0
    ), "Some visibility weights are negative, check the reduction and calibration."

    # either average the polarizations or remove the pol dimension
    npol = data.shape[0]
    if npol == 2:
        data = np.sum(data * weight[:, np.newaxis, :], axis=0) / np.sum(weight, axis=0)
        weight = np.sum(weight, axis=0)
    elif npol == 1:
        data = np.squeeze(data)
        weight = np.squeeze(weight)
    else:
        raise AssertionError("npol must be 1 or 2. Unknown value:", npol)

    # after this step,
    # data should be [3, nvis]
    # weights should be [nvis]

    # convert uu and vv to kilolambda
    uu, vv, ww = uvw  # unpack into len nvis vectors
    # broadcast to the same shape as the data
    # stub to broadcast uu,vv, and weights to all channels
    broadcast = np.ones((nchan, 1))
    uu = uu * broadcast
    vv = vv * broadcast
    weight = weight * broadcast

    # calculate wavelengths in meters
    wavelengths = c_ms / chan_freq[:, np.newaxis]  # m

    # calculate baselines in klambda
    uu = 1e-3 * uu / wavelengths  # [klambda]
    vv = 1e-3 * vv / wavelengths  # [klambda]

    frequencies = chan_freq * 1e-9  # [GHz]

    return Cube(
        frequencies,
        uu,
        vv,
        weight,
        data.real,
        data.imag,
        CASA_convention=True,
    )