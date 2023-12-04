import pandas as pd


def load_data(dir_csv):
    data = pd.read_csv(dir_csv, header=None)

    return data.to_numpy()
