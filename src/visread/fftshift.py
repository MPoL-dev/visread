import numpy as np
import casatools
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# initialize the relevant CASA tools
tb = casatools.table()
ms = casatools.ms()
msmd = casatools.msmetadata()

c_ms = 2.99792458e8  # [m s^-1]


# this routine is meant primarily as a means to do the correction for ALMA Doppler setting.
# It is designed to go from the TOPO frame to the BARY frame. This is a time-dependent and source-dependent
# conversion (it also depends on the observatory location).
# 
# If you require your velocities in a frame other than BARY, you can later apply a standard offset
# from BARY to LSRK, which is generally time and observatory independent (but still source-dependent) 
# and can be determined using tools like astropy.

# first plan of attack
# routine that will take a specific integration (set time) and apply a known velocity shift to it
# need
# * velocity shift 
# * channel frequency
# * complex data 
# what is the dimensionality of the data? It should have (npolarizations, nchan, nvis) 
# or (npolarizations, nchan, nitr, nintegrations)
# * use the Doppler formula to calculate a Delta nu shift.
# * check whether the frequency shift on a channel is linear on the first channel and on the last channel
# if it exceeds some threshold (for wide-bandwidth), we should stop.

# then, use np.fftshift along the correct dimension (user specifies nchan direction)
# then, do the FFT
# then, apply the per-Fourier component phase shift for the given delta nu
# then, do the iFFT
# then, do the np.fftshift

# larger wrapper routine that will take a list of times and calculate velocity shifts for each
# then, will go down the list and apply these shifts to the datae
