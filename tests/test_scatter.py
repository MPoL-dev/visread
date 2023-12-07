import numpy as np
from visread import scatter


def test_get_scatter(data_dict):
    scatter_XX, scatter_YY = scatter.get_scatter(
        data_dict["data"], data_dict["weight"], data_dict["flag"], data_dict["model"]
    )


def test_calculate_rescale_factor(data_dict):
    scatter_XX, scatter_YY = scatter.get_scatter(
        data_dict["data"], data_dict["weight"], data_dict["flag"], data_dict["model"]
    )

    vals = np.array(
        [
            scatter.calculate_rescale_factor(s)
            for s in [
                scatter_XX.real,
                scatter_XX.imag,
                scatter_YY.real,
                scatter_YY.imag,
            ]
        ]
    )

    print("Scatter rescales", vals)
