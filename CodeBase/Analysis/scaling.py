import numpy as np
import pandas as pd
from os import listdir

import tensorflow as tf
from pathlib import Path
from typing import Tuple
from os.path import isfile, join

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

seed = tf.random.set_seed(1)

class ScaleAndPrep:
    def __init__(self, path: str) -> None:
        self.path = path
        self.onlyfiles = self.getDfNames()

    def getDfNames(self) -> Tuple[str, ...]:
        """
        Fetches all objects in a directory

        Returns:
            Tuple[str, ...]: list of pathnames
        """
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]

    def mergeDfs(self, exlude=["data18"]) -> None:
        files = self.onlyfiles.copy()

        dfs = []
        
        for file in files:

            exl = [file.find(exl) for exl in exlude]

            df = pd.read_hdf(self.path / file)
            if sum(exl) > -1:
                self.data = df
                continue

            
            dfs.append(df)

        self.df = pd.concat(dfs)


    def scaleAndSplit(self):
        try:
            self.df
        except:
            self.mergeDfs()

        X_b_train, X_b_val = train_test_split(self.df, test_size=0.2, random_state=seed)

        weights_train = X_b_train["wgt_SG"]
        weights_val = X_b_val["wgt_SG"]

        print(len(weights_train), len(weights_val), len(weights_train) + len(weights_val))

