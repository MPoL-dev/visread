import numpy as np
import matplotlib.pyplot as plt
import visread
from visread import visualization


def test_plot_baselines(data_dict, tmp_path):
    fig = visualization.plot_baselines(
        data_dict["uu"], data_dict["vv"], title="Mock Dataset"
    )
    fig.savefig(tmp_path / "baselines.png", dpi=300)
    plt.close("all")


def test_plot_scatter_hist(data_dict, tmp_path):
    scatter_XX, scatter_YY = visread.scatter.get_scatter(
        data_dict["data"], data_dict["weight"], data_dict["flag"], data_dict["model"]
    )

    fig = visualization.scatter_hist(scatter_XX, scatter_YY)
    fig.savefig(tmp_path / "scatter_hist.png", dpi=300)
    plt.close("all")
