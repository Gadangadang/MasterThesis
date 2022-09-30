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
from scaling import *
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


class plotRMM:
    def __init__(self, path: Path, rmm_structure: dict, N_row: int):
        self.path = path
        self.rmm_structure = rmm_structure
        self.N_row = N_row
        self.onlyfiles = self.getDfNames()

    def getDfNames(self) -> Tuple[str, ...]:
        """
        Fetches all objects in a directory

        Returns:
            Tuple[str, ...]: list of pathnames
        """
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]

    def plotRMM(self):

        print("*** Plotting starting ***")

        for idx, file in enumerate(self.onlyfiles):
            file_idx = file.find("_3lep")
            df = pd.read_hdf(self.path / file)
            print(file[:file_idx])
            self.plotDfRmmMatrix(df, file[:file_idx])

        print("*** Plotting done ***")

    def plotDfRmmMatrix(self, df: pd.DataFrame, process: str) -> None:

        col = len(df.columns)
        row = len(df)

        print("")
        print(f"Size: {row}")
        print("")
        df2 = df.mean()

        tot = len(df2)
        row = int(np.sqrt(tot))
        print(row)

        rmm_mat = np.zeros((row, row))

        df2 = df2.to_numpy()

        p = 0

        for i in range(row):
            for j in range(row):
                rmm_mat[i, j] = df2[p]
                p += 1

        names = [" "]

        for i in range(1, self.N_row):
            name = self.rmm_structure[i][0]
            names.append(name)

        fig, ax = plt.subplots()

        im, cbar = self.heatmap(rmm_mat, names, names, ax=ax, cbarlabel="Intensity")
        texts = self.annotateHeatmap(im, valfmt="{x:.3f}")

        im = ax.imshow(rmm_mat)

        fig.tight_layout()

        plt.savefig(f"../../Figures/testing/rmm_avg_{process}.pdf")

    def heatmap(
        self, data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs
    ):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # Turn spines off and create white grid.
        # ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotateHeatmap(
        self,
        im,
        data=None,
        valfmt="{x:.2f}",
        textcolors=("black", "white"),
        threshold=None,
        **textkw,
    ):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.0

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center", verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts


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
