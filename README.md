# visread
A minimal tool built on CASA6 to read visibilities directly from calibrated CASA Measurement Sets

# Introduction

The purpose of this package is to provide a quick and easy `read("myFile.ms")` function to load visibilities from a calibrated CASA Measurement Set file into memory in a Python program. You may wish to plot the baseline distributions, analyze the visibilities themselves, or synthesize images from them using software like [MPoL](https://github.com/MPoL-dev).

The visibilities are returned as an object. 

Code Example 

    import visread
    vis = visread.read("myFile.ms")
    print(vis)

More information is provided in the Documentation.

Currently, there is only one primary visibility container for spectral line measurement sets, whose attributes are driven by the development needs of the MPoL project. A set of single-channel continuum visibilities is just a subset of this.

If you identify and configure an additional visibility container (i.e., polarization), pull requests are welcome.

# Lineage

The code in this package is fairly simple, but it is also built using CASA knowledge gained from a number of collaborators, who are credited in the CONTRIBUTORS.md file. You may also be interested in investigating the [`vis_sample`](https://github.com/AstroChem/vis_sample) and [`UVHDF5`](https://github.com/AstroChem/UVHDF5) packages, as they provide some similar capabilities with additional functionality.


