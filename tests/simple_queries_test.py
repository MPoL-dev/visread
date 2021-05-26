from visread import examine


def test_col_read(ms_cube_path):
    examine.get_colnames(ms_cube_path)