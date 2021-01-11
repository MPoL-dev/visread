Introduction and Measurement Set Preparation
============================================

Why does this package exist? 
----------------------------

Frequently, radio astronomers might like to work directly with the visibility data themselves to make publication-quality plots of the baseline distributions, fit visibility-plane models, or experiment with alternative imaging algorithms. `CASA <https://casa.nrao.edu/casadocs-devel/stable>`_ is the facility software reduction package for many radio telescopes (including ALMA and the VLA), and radio datasets are stored in the standard `Measurement Set format <https://casa.nrao.edu/casadocs-devel/stable/casa-fundamentals>`_. Until recently, because of the monolithic nature of CASA releases, it was very difficult for a user to directly access the visibilities inside of their own Python programs. 

The situation has vastly improved with the release of the `CASA 6.x modular installations <https://casa.nrao.edu/casadocs-devel/stable/usingcasa/obtaining-and-installing>`_, allowing a user to `pip` install the `casatools` package directly into their own Python environment. However, the series of commands needed to extract visibilities from a measurement set is still comparatively low-level and only sparsely documented. This package is meant to serve as a small piece of glue to extract the visibilities from a calibrated and processed Measurement Set and return them to the user in a well-documented format. 

Preparing your measurement set 
------------------------------

One of the great (but also very tricky) things about the Measurement Set format is that its internal structure can assume complex layouts necessary to represent the diversity of raw data formats from single-dish radio telescopes and interferometric arrays. The many choices involved with distilling this complexity down is probably one of the reasons an ``export_visibilities`` function doesn't yet exist within CASA itself.

We envision *visread* as a wrapper script for simple operations to read calibrated visibilities from a measurement set with a simple, minimal format. By that, we mean

* one spectral window 
* one target
* if the spectral window contains main channels, it has been truncated to include only those with data that you want to analyze. 
* only XX or only XX and YY linear polarizations

This means that before turning to *visread*, the user will want to spend some time with CASA familiarizing themselves with their data. If, as is usually the case, their measurement set does not meet the criteria above but contains multiple spectral windows or targets (such as calibrators), they will want to spend some time explicitly making the reduction choices themselves to get their visibilities into this minimal format. You'll want to use CASA commands like `split <https://casa.nrao.edu/casadocs/casa-5-1.2/uv-manipulation/splitting-out-calibrated-uv-data-split>`_, `mstransform <https://casa.nrao.edu/casadocs/casa-5.4.1/uv-manipulation/manipulating-visibilities-with-mstransform>`_, and `cvel2 <https://casa.nrao.edu/casadocs/casa-6.1.0/global-task-list/task_cvel2/about>`_ to first subselect those visibilities you want (possibly average them) and create a new measurement set containing only them.

Even if your are not planning to export your visibilities with *visread*, it's still nice to have a copy of your calibrated and reduced data in such a minimal format. The idea is that you could run a minimal `tclean command <https://casa.nrao.edu/casadocs/casa-6.1.0/global-task-list/task_tclean/about>`_ to produce your image or channel maps::

    tclean(vis="minimial_vis.ms",
        imagename="myImage",
        imsize=1024, 
        cell="0.02arcsec",
        specmode="cube", 
        weighting="briggs", 
        robust=0.5)

without needing to subselect portions of your measurement set using `tclean <https://casa.nrao.edu/casadocs/casa-6.1.0/global-task-list/task_tclean/about>`_ subparameters like ``field``, ``spw``, ``timerange``, ``uvrange``, ``antenna``, ``scan``, ``observation``, ``intent``, ``start``, ``width``, ``nchan``, or ``outframe``, because these transformations would already have been accomplished with `split <https://casa.nrao.edu/casadocs/casa-5-1.2/uv-manipulation/splitting-out-calibrated-uv-data-split>`_, `mstransform <https://casa.nrao.edu/casadocs/casa-5.4.1/uv-manipulation/manipulating-visibilities-with-mstransform>`_, and/or `cvel2 <https://casa.nrao.edu/casadocs/casa-6.1.0/global-task-list/task_cvel2/about>`_. Once you've created this minimal measurement set to use with *visread*, it is an **excellent** idea to first image it using `tclean <https://casa.nrao.edu/casadocs/casa-6.1.0/global-task-list/task_tclean/about>`_ to make sure it actually contains the visibilities as you wanted them processed.

The idea behind this workflow is to use the CASA facility software package for what it's designed for (processing radio interferometry data in the measurement set format) and to put any decisions about data averaging or subselection explicitly in the hands of the user (rather than incorrectly assuming what they want within *visread*). If you want to preserve some of the distincitions in the measurement set in your exported visibilities (if, for example, it was very important to preserve which spectral window each visibility came from), we would recommend making a separate minimal measurement set for each spectral window and using *visread* to read those individually.

If your data is more complicated than can be represented in our version of a minimal format, then we would recommend using the CASA6 *casatools* directly to access your data at a lower level. You may find inspiration by looking at the `source code <https://github.com/MPoL-dev/visread/blob/main/src/visread/visread.py>`_ of the ``visread.read`` function.

