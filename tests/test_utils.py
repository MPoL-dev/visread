import numpy as np
import matplotlib.pyplot as plt
from visread import utils

def test_plot_gaussian(tmp_path):
    xs = np.linspace(-4, 4, num=300)
    ys = utils.gaussian(xs)
    fig, ax = plt.subplots(nrows=1)
    ax.plot(xs, ys)
    fig.savefig(tmp_path / "gaussian.png", dpi=300)
    plt.close("all")
