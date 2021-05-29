# run from inside the tutorials directory

import numpy as np
import tarfile
import os
import requests
import casatools

tb = casatools.table()
ms = casatools.ms()

fname_tar = "AS209_continuum.ms.tar.gz"

if not os.path.exists(fname_tar):
    url = (
        "https://almascience.eso.org/almadata/lp/DSHARP/MSfiles/AS209_continuum.ms.tgz"
    )
    r = requests.get(url)

    with open(fname_tar, "wb") as f:
        f.write(r.content)

fname = "AS209_continuum.ms"

if not os.path.exists(fname):
    with tarfile.open(fname_tar) as tar:
        tar.extractall()

# Note that this process may take about 30 - 45 minutes, depending on your computing environment.
ms.open(fname)
# select the spectral window
ms.selectinit(datadescid=0)
# query the desired columnames as a list
query = ms.getdata(["MODEL_DATA"])
# always a good idea to reset the earmarked data
ms.selectinit(reset=True)
ms.close()

if len(query["model_data"]) == 0:

    # reproduce the DSHARP image using casa6
    import casatasks
    import shutil

    """ Define simple masks and clean scales for imaging """
    mask_pa = 86  # position angle of mask in degrees
    mask_maj = 1.3  # semimajor axis of mask in arcsec
    mask_min = 1.1  # semiminor axis of mask in arcsec
    mask_ra = "16h49m15.29s"
    mask_dec = "-14.22.09.04"

    common_mask = "ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]" % (
        mask_ra,
        mask_dec,
        mask_maj,
        mask_min,
        mask_pa,
    )

    imagename = "AS209"

    # clear any existing image products
    for ext in [
        ".image",
        ".mask",
        ".model",
        ".pb",
        ".psf",
        ".residual",
        ".sumwt",
        ".image.pbcor",
    ]:
        obj = imagename + ext
        if os.path.exists(obj):
            shutil.rmtree(obj)

    casatasks.delmod(vis=fname)

    casatasks.tclean(
        vis=fname,
        imagename=imagename,
        specmode="mfs",
        deconvolver="multiscale",
        scales=[0, 5, 30, 100, 200],
        weighting="briggs",
        robust=-0.5,
        gain=0.2,
        imsize=3000,
        cell=".003arcsec",
        niter=50000,
        threshold="0.08mJy",
        cycleniter=300,
        cyclefactor=1,
        uvtaper=[".037arcsec", ".01arcsec", "162deg"],
        mask=common_mask,
        nterms=1,
        savemodel="modelcolumn",
    )

    if os.path.exists(imagename + ".fits"):
        os.remove(imagename + ".fits")
    casatasks.exportfits(
        imagename + ".image", imagename + ".fits", dropdeg=True, dropstokes=True
    )
