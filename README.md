# visreader
Small tools built on CASA6 to read visibilities directly from calibrated CASA Measurement Sets

# Introduction

The purpose of this package is to provide a quick and easy `read("myFile.ms")` function to load visibilities from a calibrated CASA Measurement Set file into memory in a Python program. You may wish to analyze them, plot the baseline distributions, or synthesize images from them using software like [MPoL](https://github.com/MPoL-dev).

The visibilities are returned as an object. Currently, there is only one primary visibility container specialized for the purposes of the MPoL project. If you identify a use case for a different visibility container (i.e., polarization), please submit a pull request.

# Lineage

The code in this package is fairly simple, but it is also built using CASA knowledge gained from scripts provided by a number of collaborators, who are credited in the CONTRIBUTORS.md file.


