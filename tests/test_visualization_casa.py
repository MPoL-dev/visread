import matplotlib.pyplot as plt
import pytest

# ascertain if casatasks is installed
try:
    from visread import visualization_casa
    no_casa = False
except ModuleNotFoundError:
    no_casa = True

@pytest.mark.skipif(no_casa, reason="modular casa not available on this system")
def test_plot_baselines(ms_cont_path, tmp_path):
    fig = visualization_casa.plot_baselines(ms_cont_path, 0)
    fig.savefig(tmp_path / "baselines_casa.png", dpi=300)
    plt.close("all")
