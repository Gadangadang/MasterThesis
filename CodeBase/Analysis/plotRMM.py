import time
import random
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from os import listdir
from numpy import array
import tensorflow as tf
import keras_tuner as kt
from pathlib import Path
from typing import Tuple
import plotly.express as px
import matplotlib.pyplot as plt
from os.path import isfile, join
from sklearn.compose import ColumnTransformer
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Utilities.config import *
from Utilities.pathfile import *



seed = tf.random.set_seed(1)

tf.keras.utils.get_custom_objects()["leaky_relu"] = tf.keras.layers.LeakyReLU()


scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}
scaler = scalers[SCALER]

if SMALL:
    arc = "small"
else:
    arc = "big"

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
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]  # type: ignore

    def plotRMM(self):
        """_summary_"""

        print("*** Plotting starting ***")

        for idx, file in enumerate(self.onlyfiles):
            file_idx = file.find("_3lep")
            df = pd.read_hdf(self.path / file)
            print(file[:file_idx])
            self.plotDfRmmMatrix(df, file[:file_idx])

        print("*** Plotting done ***")

    def plotDfRmmMatrix(self, df: pd.DataFrame, process: str) -> None:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            process (str): _description_
        """

        col = len(df.columns)
        row = len(df)

        df2 = df.mean()

        tot = len(df2)
        row = int(np.sqrt(tot))

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

        # rmm_mat[rmm_mat < 0.00009] = np.nan

        rmm_mat[rmm_mat == 0] = np.nan

        fig = px.imshow(
            rmm_mat,
            labels=dict(x="Particles", y="Particles", color="Intensity"),
            x=names,
            y=names,
            aspect="auto",
            color_continuous_scale="Viridis",
            text_auto=".3f",
        )
        fig.update_xaxes(side="top")

        fig.write_image(f"../../Figures/testing/rmm/rmm_avg_{process}.pdf")

    def plotDfRmmMatrixNoMean(self, df: pd.DataFrame, process: str, idx: int, additional_info="", fake=False) -> None:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            process (str): _description_
            idx (int): _description_
            fake (bool, optional): _description_. Defaults to False.
        """

        try:  # In case pandas dataframe is passed
            col = len(df.columns)
            df2 = df.iloc[idx].to_numpy()
        except:  # In case regular numpy array is passed
            col = len(df)
            df2 = df[idx]

        row = len(df)

        tot = len(df2)
        row = int(np.sqrt(tot))
        
        print(row)

        rmm_mat = np.zeros((row, row))

        p = 0

        for i in range(row):
            for j in range(row):
                rmm_mat[i, j] = df2[p]
                p += 1

        names = [" "]

        for i in range(1, self.N_row):
            name = self.rmm_structure[i][0]
            number = name[-1]
            part_type = name[:-2]
            name = rf"${part_type}_{number}$"
            
            names.append(name)
            
        

        # rmm_mat[rmm_mat < 0.00009] = np.nan

        rmm_mat[rmm_mat == 0.] = np.nan

        fig = px.imshow(
            rmm_mat,
            labels=dict(x="Particles", y="Particles", color="Intensity"),
            x=names,
            y=names,
            aspect="auto",
            color_continuous_scale="Viridis",
            text_auto=".3f",
            title=f"Event {idx} channel {process}",
        )
        fig.update_xaxes(side="top")
        if fake:
            if additional_info:
                fig.write_image(f"../../Figures/testing/rmm/rmm_event_{idx}_{process}_{additional_info}_fake_event.pdf")
            else:
               fig.write_image(f"../../Figures/testing/rmm/rmm_event_{idx}_{process}_fake_event.pdf") 
        else:
            fig.write_image(f"../../Figures/testing/rmm/rmm_event_{idx}_{process}.pdf")

