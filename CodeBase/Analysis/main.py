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
from CodeBase.Analysis.analysis import *
from alterdataframes import *

check_cols = [
    "ele3_pt",
    "ele3_eta",
    "ele3_phi",
    "ele3_m",
    "muo3_pt",
    "muo3_eta",
    "muo3_phi",
    "muo3_m",
]

test = [
    "m_T_ele_2",
    "m_T_muo_2",
    "m_jet_0_ele_2",
    "m_jet_0_muo_2",
    "m_jet_1_ele_2",
    "m_jet_1_muo_0",
    "m_jet_1_muo_2",
    "m_ele_0_ele_2",
    "m_ele_0_muo_0",
    "m_ele_0_muo_2",
    "m_ele_1_ele_2",
    "m_ele_1_muo_2",
    "m_ele_2_muo_0",
    "m_ele_2_muo_1",
    "m_ele_2_muo_2",
    "m_muo_0_muo_2",
    "m_muo_1_muo_2",
]

rmm_structure = {
    1: [
        "jet_0",
        "jetPt[jet_SG]",
        "jetEta[jet_SG]",
        "jetPhi[jet_SG]",
        "jetM[jet_SG]",
        0,
    ],
    2: [
        "jet_1",
        "jetPt[jet_SG]",
        "jetEta[jet_SG]",
        "jetPhi[jet_SG]",
        "jetM[jet_SG]",
        1,
    ],
    3: [
        "ele_0",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        0,
    ],
    4: [
        "ele_1",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        1,
    ],
    5: [
        "ele_2",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        2,
    ],
    6: [
        "muo_0",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        0,
    ],
    7: [
        "muo_1",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        1,
    ],
    8: [
        "muo_2",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        2,
    ],
}


def plot_checks(df: pd.DataFrame, nonzero=False):

    idxs = []

    for col in test:

        idx = np.where(df[col] > 5000)[0]
        if len(idx) < 1:
            pass
        else:
            for i in idx:
                # print(col, i, df[col][i])

                idxs.append((col, i, df[col][i]))

    for col in check_cols:

        for check, i, val in idxs:
            if nonzero:
                if df[col][i] > 0:
                    print(
                        f"Var: {check}, Var_val: {val}, Row: {i}, Property: {col}, Prop_val: {df[col][i]}"
                    )
            else:
                print(
                    f"Var: {check}, Var_val: {val}, Row: {i}, Property: {col}, Prop_val: {df[col][i]}"
                )

        plt.plot(df[col])
        # plt.ylim(-10, 10)

        plt.savefig(testing_images / f"{col}.pdf")


def test_vals(choice=False):
    names = [f for f in listdir(path) if isfile(join(path, f))]

    for file in names:
        id = file.find("_3lep")
        print(file[:id])
        df = pd.read_hdf(path / file)

        plot_checks(df, choice)

        print(" ")
        print(" ")
        print(" ")

    """
    col = "m_ele_0_muo_2"
    val = df[(df[col] > 100)  ][col]
    plt.scatter(range(len(val)), val)
        #plt.ylim(-10, 10)
    plt.savefig(testing_images/f"{col}.pdf" )
    """





if __name__ == "__main__":

    """ Actual analysis """
    N_j = 2
    N_l = 6

    N_col = N_j + N_l + 1
    N_row = N_col

    path = Path("/storage/William_Sakarias/Sakarias_Data")

    testing_images = Path("../../Figures/testing/")

    # exclude = ["data18"]
    # scaleandprep = ScaleAndPrep(path)

    test_vals(choice=False)

    """
    data = Data(path)
    data.alterOverflowValues()

    plo = plotRMM(path, rmm_structure, N_row)
    plo.plotRMM()

    """
    # scaleandprep.mergeDfs(exclude)

    # scaleandprep.scaleAndSplit()

    # back_df = scaleandprep.df
    # data = scaleandprep.data
