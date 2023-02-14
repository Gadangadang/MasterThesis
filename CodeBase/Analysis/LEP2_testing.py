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

from plotRMM import plotRMM
from Utilities.config import *
from Utilities.pathfile import *

seed = tf.random.set_seed(1)


scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}
scaler = scalers[SCALER]

if SMALL:
    arc = "small"
else:
    arc = "big"
    
    
class LEP2ScaleAndPrep:
    def __init__(self, path: Path, event_rmm=False, save=False, load=False, lep=3) -> None:
        """_summary_

        Args:
            path (str): _description_
            event_rmm (bool, optional):
        """
        self.path = path
        self.lep = lep
        self.onlyfiles = self.getDfNames()
        self.event_rmm = event_rmm
        self.load = load
        self.save = save
        
        # self.scaleAndSplit()
        

        

    def getDfNames(self) -> Tuple[str, ...]:
        """
        Fetches all objects in a directory

        Returns:
            Tuple[str, ...]: list of pathnames
        """
        
        if self.lep == 3:
            files = [
                f
                for f in listdir(self.path)
                if isfile(join(self.path, f))
                and f[-4:] != ".npy"
                and f[-4:] != ".csv"
                and f[-5:] != "_b.h5"
                and f[-4:] != ".txt"
                and f[-3:] != ".h5"
                and f[0:3] != "two"
            ]
        elif self.lep == 2:
            files = [
                f
                for f in listdir(self.path)
                if isfile(join(self.path, f))
                and f[-4:] != ".npy"
                and f[-4:] != ".csv"
                and f[-5:] != "_b.h5"
                and f[-4:] != ".txt"
                and f[-3:] != ".h5"
                and f[0:3] == "two"
            ]
        
      
        return files  # type: ignore
    
    def checkSizes(self):
        
        for file in self.onlyfiles:
            df = pd.read_hdf(self.path/file)
            print(df.info(memory_usage="deep"))
    

if __name__ == "__main__":
    L2 = LEP2ScaleAndPrep(DATA_PATH, True, SAVE_VAR, LOAD_VAR, lep=2)
    print(L2.onlyfiles)
    L2.checkSizes()