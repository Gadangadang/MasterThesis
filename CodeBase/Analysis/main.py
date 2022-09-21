from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from AE_tune import *
from scaling import ScaleAndPrep






if __name__ == "__main__":
    
    path = Path("/storage/William_Sakarias/Sakarias_Data")

    exclude = ["data18"]
    scaleandprep = ScaleAndPrep(path)

    df = pd.read_hdf(path/"Wjets_3lep_df_forML_bkg_signal_fromRDF.hdf5")

    plt.plot((df["m_T_ele_2"]))
    plt.savefig("test.pdf")

    """scaleandprep.mergeDfs(exclude)

    scaleandprep.scaleAndSplit()
    
    back_df = scaleandprep.df
    data = scaleandprep.data

    print(back_df["e_T_lep_0"])

    print(np.shape(back_df))

    print(np.shape(data))"""

