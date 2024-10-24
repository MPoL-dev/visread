import numpy as np
from . import scatter

try:
    import casatools

    # initialize the relevant CASA tools
    msmd = casatools.msmetadata()
    ms = casatools.ms()
except ModuleNotFoundError as e:
    print(
        "casatools module not found on system. If your system configuration is compatible, you can try installing these optional dependencies with `pip install 'visread[casa]'`. More information on Modular CASA can be found https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Modular-Packages "
    )
    raise e


def get_scatter_datadescid(
    filename,
    datadescid,
    sigma_rescale=1.0,
    apply_flags=True,
    residual=True,
    datacolumn="corrected_data",
):
    r"""
    Calculate the residuals for each polarization (XX, YY) in units of :math:`\sigma`, where

    .. math::

        \sigma = \mathrm{sigma\_rescale} \times \sigma_0

    and :math:`\sigma_0 = \sqrt{1/w}`. The scatter is defined as

    .. math::

        \mathrm{scatter} = \frac{\mathrm{DATA} - \mathrm{MODEL\_DATA}}{\sigma}

    Args:
        filename (string): measurement set filename
        datadescid (int): the DATA_DESC_ID to be queried
        sigma_rescale (int):  multiply the uncertainties by this factor
        apply_flags (bool): calculate the scatter *after* the flags have been applied
        residual (bool): if True, subtract MODEL_DATA column (from a tclean model, most likely) to plot scatter of residual visibilities.
        datacolumn (string): which datacolumn to use (i.e., 'corrected_data' or 'data'). 
        If 'corrected_data' is not available, will fall back to 'data'.

    Returns:
        scatter_XX, scatter_YY: a 2-tuple of numpy arrays containing the scatter in each polarization.
        If ``apply_flags==True``, each array will be 1-dimensional. If ``apply_flags==False``, each array
        will retain its original shape, including channelization (e.g., shape ``nchan,nvis``).

    """

    # see which keys are available
    tb = casatools.table()
    tb.open(filename)
    colnames = tb.colnames()
    tb.close()

    if datacolumn == "corrected_data":
        if datacolumn.upper() not in colnames:
            datacolumn = "data"

    ms.open(filename)
    ms.selectinit(datadescid=datadescid)
    keys = ["weight", "flag", datacolumn]
    if residual:
        keys += ["model_data"]
    q = ms.getdata(keys)
    ms.selectinit(reset=True)
    ms.close()

    if residual:
        assert (
            len(q["model_data"]) > 0
        ), "MODEL_DATA column empty, retry tclean with savemodel='modelcolumn'"
        model = q["model_data"]

    else:
        model = None

    return scatter.get_scatter(
        q[datacolumn],
        q["weight"],
        q["flag"],
        model=model,
        sigma_rescale=sigma_rescale,
        apply_flags=apply_flags,
        residual=residual,
    )


def get_sigma_rescale_datadescid(filename, datadescid, datacolumn="corrected_data"):
    """
    For a given datadescid, calculate the residual scatter in each of the XX and YY polarization visibilities, then calculate the sigma rescale factor for each of the real and imaginary values of the polarizations. Return the average of all four quantities as the final sigma rescale factor for that datadescid.

    Args:
        filename (string): path to measurement set
        datadescid (int): the spectral window in the measurement set

    Returns:
        float: the multiplicative factor by which to scale :math:`\sigma`
    """
    scatter_XX, scatter_YY = get_scatter_datadescid(
        filename, datadescid, apply_flags=True, datacolumn=datacolumn
    )

    vals = np.array(
        [
            scatter.calculate_rescale_factor(scat)
            for scat in [
                scatter_XX.real,
                scatter_XX.imag,
                scatter_YY.real,
                scatter_YY.imag,
            ]
        ]
    )

    return np.average(vals)
