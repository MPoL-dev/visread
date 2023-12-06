import numpy as np 
import matplotlib.pyplot as plt
import casatools 
from . import utils, scatter, process, visualization

def plot_baselines(filename, ddid):
    ms.open(filename)
    ms.selectinit(datadescid=datadescid)
    q = ms.getdata(["uvw"])
    ms.selectinit(reset=True)
    ms.close()

    u, v, w = q["uvw"] * 1e-3  # [km]

    return visualization.plot_baselines(u, v, "DATA_DESC_ID: {:}".format(ddid))


def plot_scatter_datadescid(
    filename,
    datadescid,
    log=False,
    sigma_rescale=1.0,
    chan_slice=None,
    apply_flags=True,
    residual=True,
    datacolumn="corrected_data"
):
    r"""
    Plot a set of histograms of the scatter of the residual visibilities for the real and
    imaginary components of the XX and YY polarizations.

    Args:
        filename (string): measurement set filename
        datadescid (int): the DATA_DESC_ID to be queried
        log (bool): should the histogram values be log scaled?
        sigma_rescale (int):  multiply the uncertainties by this factor
        chan_slice (slice): if not None, a slice object specifying the channels to subselect
        apply_flags (bool): calculate the scatter *after* the flags have been applied
        residual (bool): if True, subtract MODEL_DATA column (from a tclean model, most likely) to plot scatter of residual visibilities.
        datacolumn (string): which datacolumn to use (i.e., 'corrected_data' or 'data').

    Returns:
        matplotlib figure with scatter histograms
    """

    if chan_slice is not None:
        print("apply_flags setting is ignored when chan_slice is not None")

        scatter_XX, scatter_YY = scatter.get_scatter_datadescid(
            filename, datadescid, sigma_rescale, apply_flags=False, residual=residual, datacolumn=datacolumn
        )

        scatter_XX = scatter_XX[chan_slice]
        scatter_YY = scatter_YY[chan_slice]

    else:
        scatter_XX, scatter_YY = scatter.get_scatter_datadescid(
            filename, datadescid, sigma_rescale, apply_flags=apply_flags, residual=residual, datacolumn=datacolumn
        )

    scatter_XX = scatter_XX.flatten()
    scatter_YY = scatter_YY.flatten()

    fig = visualization.scatter_hist(scatter_XX, scatter_YY, log=log)
    fig.suptitle("DATA_DESC_ID: {:}".format(datadescid))

    return fig


def plot_weight_hist(filename, datadescid, log=False, **kwargs):

    ms.open(filename)
    ms.selectinit(datadescid=datadescid)
    q = ms.getdata(["weight"])
    ms.selectinit(reset=True)
    ms.close()

    weight_XX, weight_YY = q["weight"]

    return visualization.plot_weight_hist(weight_XX, weight_YY, "DATA_DESC_ID: {:}".format(datadescid))