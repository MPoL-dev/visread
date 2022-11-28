# meant to be run from inside docs/ directory as part
# of the Makefile process

import tarfile
import os
import requests
import casatools
import casatasks
import shutil


tb = casatools.table()
ms = casatools.ms()

homedir = os.getcwd()
workdir = "tutorials/AS209_MS"

print("homedir is ", homedir)

if not os.path.exists(workdir):
    os.makedirs(workdir)

os.chdir(workdir)

fname_tar = "AS209_continuum.ms.tar.gz"

if not os.path.exists(fname_tar):
    print("downloading tarball")
    url = (
        "https://almascience.eso.org/almadata/lp/DSHARP/MSfiles/AS209_continuum.ms.tgz"
    )
    r = requests.get(url)

    with open(fname_tar, "wb") as f:
        f.write(r.content)

fname = "AS209_continuum.ms"

if not os.path.exists(fname):
    print("extracting tarball")
    with tarfile.open(fname_tar) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)

# Check to see whether the MODEL_DATA column contains visibilities.
# If not, run the DSHARP tclean scripts.

ms.open(fname)
# select the spectral window
ms.selectinit(datadescid=0)
# query the desired columnames as a list
query = ms.getdata(["MODEL_DATA"])
# always a good idea to reset the earmarked data
ms.selectinit(reset=True)
ms.close()

if len(query["model_data"]) == 0:
    print("cleaning MS")
    # Note that this process may take about 30 - 45 minutes, depending on your computing environment.

    # reproduce the DSHARP image using casa6
    # The full reduction scripts are available online at
    # https://almascience.eso.org/almadata/lp/DSHARP/scripts/AS209_continuum.py
    # Here we just reproduce the relevant ``tclean`` commands used to produce a FITS image
    # from the final, calibrated measurement set.

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

else:
    print("MS already cleaned")

os.chdir(homedir)