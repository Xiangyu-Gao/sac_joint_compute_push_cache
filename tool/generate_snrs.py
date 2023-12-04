import os, sys, inspect
# set parent directory as sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np


if __name__ == '__main__':
    sample_len = 1200000 * 4
    # # generate all one list
    # output = [1 for i in range(sample_len)]
    # pd.DataFrame(np.array(output)).to_csv("./data/one_snrs.csv", header=False, index=False)

    # generate dynamic snrs
    snrs = [1, 2, 0.5, 3]
    output = []
    for snr in snrs:
        output += [snr] * 1200000
    pd.DataFrame(np.array(output)).to_csv("./data/dynamic_snrs.csv", header=False, index=False)
