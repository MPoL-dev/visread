import numpy as np
from visread import process


def test_convert_baselines(data_dict):
    uu = data_dict["uu"]
    freq = data_dict["freq"][0]

    process.convert_baselines(uu, freq)


def test_broadcast_and_convert(data_dict):
    uu = data_dict["uu"]
    vv = data_dict["vv"]
    freq = data_dict["freq"]

    process.broadcast_and_convert_baselines(uu, vv, freq)