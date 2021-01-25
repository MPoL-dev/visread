import numpy as np
import visread

import pytest


def test_create_Cube():
    nchan = 10
    nvis = 400
    freqs = np.linspace(4, 3, num=nchan)
    uu = np.ones((nchan, nvis))
    vv = np.ones((nchan, nvis))
    weight = np.ones((nchan, nvis))
    data_re = np.ones((nchan, nvis))
    data_im = np.ones((nchan, nvis))

    vis = visread.Cube(freqs, uu, vv, weight, data_re, data_im)


def test_create_Cube_weight_asserts():
    nchan = 10
    nvis = 400
    freqs = np.linspace(4, 3, num=nchan)
    uu = np.ones((nchan, nvis))
    vv = np.ones((nchan, nvis))
    weight = np.ones((nchan, nvis))
    data_re = np.ones((nchan, nvis))
    data_im = np.ones((nchan, nvis))

    with pytest.raises(AssertionError):
        vis = visread.Cube(freqs, uu, vv, -weight, data_re, data_im)


def test_create_Cube_shape_asserts():
    nchan = 10
    nvis = 400
    freqs = np.linspace(4, 3, num=nchan)
    uu = np.ones((nchan, nvis))
    vv = np.ones((nchan - 1, nvis))
    weight = np.ones((nchan - 2, nvis))
    data_re = np.ones((nchan, nvis))
    data_im = np.ones((nchan, nvis))

    with pytest.raises(AssertionError):
        vis = visread.Cube(freqs, uu, vv, weight, data_re, data_im)

    with pytest.raises(AssertionError):
        vis = visread.Cube(freqs[np.newaxis, :], uu, vv, weight, data_re, data_im)


def test_Cube_freq_assert():
    nchan = 10
    nvis = 400
    freqs_inc = np.linspace(3, 4, num=nchan)
    uu = np.ones((nchan, nvis))
    vv = np.ones((nchan - 1, nvis))
    weight = np.ones((nchan - 2, nvis))
    data_re = np.ones((nchan, nvis))
    data_im = np.ones((nchan, nvis))

    with pytest.raises(AssertionError):
        vis = visread.Cube(freqs_inc, uu, vv, weight, data_re, data_im)


def test_Cube_dtypes():
    nchan = 10
    nvis = 400
    freqs = np.linspace(4, 3, num=nchan)
    uu = np.ones((nchan, nvis))
    vv = np.ones((nchan, nvis))
    weight = np.ones((nchan, nvis))
    data_re = np.ones((nchan, nvis))
    data_im = np.ones((nchan, nvis))

    data_re_bogus = np.ones((nchan, nvis), dtype=np.complex128)
    data_im_bogus = np.ones((nchan, nvis), dtype=np.complex128)

    with pytest.raises(AssertionError):
        vis = visread.Cube(freqs, uu, vv, weight, data_re_bogus, data_im)

    with pytest.raises(AssertionError):
        vis = visread.Cube(freqs, uu, vv, weight, data_re, data_im_bogus)
