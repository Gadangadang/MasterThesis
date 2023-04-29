import re
import os
import time
import random
import requests
import numpy as np
import polars as pl
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
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from plotRMM import plotRMM
from histo import PlotHistogram

from Utilities.config import *
from Utilities.pathfile import *

seed = tf.random.set_seed(1)


scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}
scaler = scalers[SCALER]

if SMALL:
    arc = "small"
else:
    arc = "big"


class DataHandling:
    def __init__(
        self, path: Path, event_rmm=False, save=False, load=False, lep=3, convert=True
    ) -> None:
        """_summary_

        Args:
            path (Path): _description_
            event_rmm (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
            load (bool, optional): _description_. Defaults to False.
            lep (int, optional): _description_. Defaults to 3.
            convert (bool, optional): _description_. Defaults to True.
        """

        self.path = path
        self.lep = lep
        self.onlyfiles = self.getDfNames()
        self.event_rmm = event_rmm
        self.load = load
        self.save = save
        self.convert = convert
        # self.scaleAndSplit()
        self.totmegasets = 10

        self.epochs = EPOCHS
        self.b_size = BACTH_SIZE

        self.channels = [
            "Zttjets",
            "Wjets",
            "singletop",
            "ttbar",
            "Zeejets",
            "Zmmjets",
            "Diboson",
        ]

        self.parqs = [
            f
            for f in listdir(self.path)
            if isfile(join(self.path, f)) and f[-8:] == ".parquet"
        ]

        if SMALL:
            self.checkpointname = "small"
        else:
            self.checkpointname = "big"
            
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
                and f[-8:] != ".parquet"
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
                and f[-8:] != ".parquet"
            ]

        return files  # type: ignore

    def convertParquet(self):
        """
        Converts all hdf5 files to parquet files for use of polars later
        """

        if self.convert:

            for file in self.onlyfiles:

                print(f"Converting {file}")

                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
                with strategy.scope():
                    start = file.find("two_")
                    stop = file.find("_3lep")
                    name = file[start + 4 : stop]
                    if name not in ["data1516", "data15", "data16"]:
                        continue

                    df = pd.read_hdf(self.path / file)

                    # scaled_df = scaler.fit_transform(df)
                    name = "twolep_" + name + ".parquet"

                    df.to_parquet(self.path / name)
                    print(f"{name} done")
                    print(" ")

        self.parqs = [
            f
            for f in listdir(self.path)
            if isfile(join(self.path, f)) and f[-8:] == ".parquet"
        ]

        print(self.parqs)

    def createMCSubsamples(self):
        """
        Create subsets that maintains the SM MC distribution.

        """

        try:
            self.parqs
        except:
            self.convertParquet()

        print("Subsampling started")
        for file in self.parqs:

            start = file.find("twolep_")
            end = file.find(".parquet")

            name = file[start + 7 : end]

            print(f"Subsampling channel: {name}")

            df = pl.read_parquet(self.path / file, use_pyarrow=True)

            df = df.drop(
                [
                    "nlep_BL",
                    "nlep_SG",
                    "ele_0_charge",
                    "ele_1_charge",
                    "ele_2_charge",
                    "muo_0_charge",
                    "muo_1_charge",
                    "muo_2_charge",
                    "TrileptonMass",
                ],
            )

            x_b_train, x_b_val = train_test_split(df, test_size=0.2, random_state=seed)

            self.sampleSet(x_b_train, x_b_val, name)

        print("Subsampling done")

    def sampleSet(self, xtrain, xval, name):

        count1 = len(xtrain)
        count2 = len(xval)
        names_train = [name] * count1
        names_train = np.asarray(names_train)

        names_val = [name] * count2
        names_val = np.asarray(names_val)

        # * Sample from training and validation set
        indices_train = np.asarray(range(len(xtrain)))
        np.random.shuffle(indices_train)
        split_idx_train = np.array_split(indices_train, self.totmegasets)

        indices_val = np.asarray(range(len(xval)))
        np.random.shuffle(indices_val)
        split_idx_val = np.array_split(indices_val, self.totmegasets)

        weights_train_tot = xtrain["wgt_SG"].to_numpy()
        weights_val_tot = xval["wgt_SG"].to_numpy()

        if name == "data1516":
            isBSM_train = xtrain["isBSM"].to_numpy()
            isBSM_val = xval["isBSM"].to_numpy()
            xtrain = xtrain.drop(
                ["flcomp", "wgt_SG", "isBSM"],
            )
            xval = xval.drop(
                ["flcomp", "wgt_SG", "isBSM"],
            )
        else:
            xtrain = xtrain.drop(
                ["flcomp", "wgt_SG"],
            )
            xval = xval.drop(
                ["flcomp", "wgt_SG"],
            )

        etmiss_val = xval["e_T_miss"].to_numpy()
        etmiss_train = xtrain["e_T_miss"].to_numpy()

        megaset = 0
        for idx_set_train, idx_set_val in zip(split_idx_train, split_idx_val):
            if name in ["data17", "data18"]:
                break
            print(f"name: {name}; megaset: {megaset}")

            weights_train = weights_train_tot[idx_set_train]
            weights_val = weights_val_tot[idx_set_val]

            train_categories = names_train[idx_set_train]
            val_categories = names_val[idx_set_val]

            # * Save weights, categories
            np.save(
                DATA_PATH
                / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_weights_train",
                weights_train,
            )
            np.save(
                DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_weights_val",
                weights_val,
            )

            np.save(
                DATA_PATH
                / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_categories_train",
                train_categories,
            )
            np.save(
                DATA_PATH
                / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_categories_val",
                val_categories,
            )

            # * Save the actual dataframe
            np.save(
                DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_x_train",
                xtrain.to_numpy()[idx_set_train],
            )
            np.save(
                DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_x_val",
                xval.to_numpy()[idx_set_val],
            )

            np.save(
                DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_etmiss_val",
                etmiss_val[idx_set_val],
            )

            # * Save etmiss, isBSM
            np.save(
                DATA_PATH
                / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_etmiss_train",
                etmiss_train[idx_set_train],
            )

            if name == "data1516":

                np.save(
                    DATA_PATH
                    / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_isBSM_train",
                    isBSM_train[idx_set_train],
                )

                np.save(
                    DATA_PATH
                    / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_isBSM_val",
                    isBSM_val[idx_set_val],
                )

            megaset += 1

    def mergeMegaBatches(self):

        print(" ")
        print(f"Megabatch merging started")

        for megaset in range(self.totmegasets):
            print(f"Running merging on megabatch: {megaset}")
            FETCH_PATH = DATA_PATH / f"Megabatches/MB{megaset}"
            MERGE_PATH = FETCH_PATH / f"MergedMB{megaset}"
            
            
            

            xval_etmiss = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "etmiss_val" in filename
                and ("data1516" not in filename
                and "data15" not in filename
                and "data16" not in filename)
            ]

            xtrain_etmiss = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "etmiss_train" in filename
                and ("data1516" not in filename
                and "data15" not in filename
                and "data16" not in filename)
            ]

            xtrain_etmiss = np.concatenate((xtrain_etmiss), axis=0)

            xval_etmiss = np.concatenate((xval_etmiss), axis=0)

            xtrains = [
                np.load(FETCH_PATH / filename)[:, :-1]
                for filename in os.listdir(FETCH_PATH)
                if "x_train" in filename
                and ("data1516" not in filename
                and "data15" not in filename
                and "data16" not in filename)
            ]
            xtrain_weights = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "weights_train" in filename
                and ("data1516" not in filename
                and "data15" not in filename
                and "data16" not in filename)
            ]
            xtrain_categories = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "categories_train" in filename
                and ("data1516" not in filename
                and "data15" not in filename
                and "data16" not in filename)
            ]

            xvals = [
                np.load(FETCH_PATH / filename)[:, :-1]
                for filename in os.listdir(FETCH_PATH)
                if "x_val" in filename
                and ("data1516" not in filename
                and "data15" not in filename
                and "data16" not in filename)
            ]
            xval_weights = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "weights_val" in filename
                and ("data1516" not in filename
                and "data15" not in filename
                and "data16" not in filename)
            ]
            xval_categories = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "categories_val" in filename
                and ("data1516" not in filename
                and "data15" not in filename
                and "data16" not in filename)
            ]
            
            xtrain = np.concatenate((xtrains), axis=0)
            x_train_cats = np.concatenate((xtrain_categories), axis=0)
            x_train_weights = np.concatenate((xtrain_weights), axis=0)

            

            xval = np.concatenate((xvals), axis=0)
            x_val_cats = np.concatenate((xval_categories), axis=0)
            x_val_weights = np.concatenate((xval_weights), axis=0)

            
            print("SM MC partitioning and concatenation done")

            #* Data 15 and 16
            data15_and_16_xval_etmiss = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "etmiss_val" in filename
                and ("data15" in filename
                or "data16" in filename)
            ]

            data15_and_16_xtrain_etmiss = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "etmiss_train" in filename
                and ("data15" in filename
                or "data16" in filename)
            ]
            
            data15_and_16_xtrain_etmiss = np.concatenate(
                (data15_and_16_xtrain_etmiss[:2]), axis=0
            )
            data15_and_16_xval_etmiss = np.concatenate(
                (data15_and_16_xval_etmiss[:2]), axis=0
            )
            data15_and_16_etmiss = np.concatenate(
                (data15_and_16_xtrain_etmiss, data15_and_16_xval_etmiss), axis=0
            )

            data15_and_16_xtrains = [
                np.load(FETCH_PATH / filename)[:, :-1]
                for filename in os.listdir(FETCH_PATH)
                if "x_train" in filename
                and ("data15" in filename
                or "data16" in filename)
            ]
            data15_and_16_xtrain_weights = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "weights_train" in filename
                and ("data15" in filename
                or "data16" in filename)
            ]
            data15_and_16_xtrain_categories = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "categories_train" in filename
                and ("data15" in filename
                or "data16" in filename)
            ]

            data15_and_16_xvals = [
                np.load(FETCH_PATH / filename)[:, :-1]
                for filename in os.listdir(FETCH_PATH)
                if "x_val" in filename 
                and ("data15" in filename 
                or "data16" in filename)
            ]
            data15_and_16_xval_weights = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "weights_val" in filename
                and ("data15" in filename
                or "data16" in filename)
            ]
            data15_and_16_xval_categories = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "categories_val" in filename
                and ("data15" in filename
                or "data16" in filename)
            ]

            data15_and_16_xtrain = np.concatenate((data15_and_16_xtrains[:2]), axis=0)
            data15_and_16_x_train_cats = np.concatenate(
                (data15_and_16_xtrain_categories[:2]), axis=0
            )
            data15_and_16_x_train_weights = np.concatenate(
                (data15_and_16_xtrain_weights[:2]), axis=0
            )

            data15_and_16_xval = np.concatenate((data15_and_16_xvals[:2]), axis=0)
            data15_and_16_x_val_cats = np.concatenate(
                (data15_and_16_xval_categories[:2]), axis=0
            )
            data15_and_16_x_val_weights = np.concatenate(
                (data15_and_16_xval_weights[:2]), axis=0
            )

            data15_and_16_x_weights = np.concatenate(
                (data15_and_16_x_train_weights, data15_and_16_x_val_weights), axis=0
            )
            data15_and_16_x_cats = np.concatenate(
                (data15_and_16_x_train_cats, data15_and_16_x_val_cats), axis=0
            )
            data15_and_16 = np.concatenate(
                (data15_and_16_xtrain, data15_and_16_xval), axis=0
            )
            
            print("Data 15 and data 16 partitioning and concatenation done")

            # * Data 1516 mix
            data1516_xval_etmiss = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "etmiss_val" in filename and "data1516" in filename
            ]
            data1516_xval_isBSM = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "isBSM_val" in filename and "data1516" in filename
            ]

            data1516_xtrain_isBSM = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "isBSM_train" in filename and "data1516" in filename
            ]
            data1516_xtrain_isBSM = np.concatenate((data1516_xtrain_isBSM), axis=0)
            data1516_xval_isBSM = np.concatenate((data1516_xval_isBSM), axis=0)
            data1516_isBSM = np.concatenate(
                (data1516_xtrain_isBSM, data1516_xval_isBSM), axis=0
            )

            data1516_xtrain_etmiss = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "etmiss_train" in filename and "data1516" in filename
            ]

            data1516_xtrain_etmiss = np.concatenate((data1516_xtrain_etmiss), axis=0)
            data1516_xval_etmiss = np.concatenate((data1516_xval_etmiss), axis=0)
            data1516_etmiss = np.concatenate(
                (data1516_xtrain_etmiss, data1516_xval_etmiss), axis=0
            )

            data1516_xtrains = [
                np.load(FETCH_PATH / filename)[:, :-1]
                for filename in os.listdir(FETCH_PATH)
                if "x_train" in filename and "data1516" in filename
            ]
            data1516_xtrain_weights = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "weights_train" in filename and "data1516" in filename
            ]
            data1516_xtrain_categories = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "categories_train" in filename and "data1516" in filename
            ]

            data1516_xvals = [
                np.load(FETCH_PATH / filename)[:, :-1]
                for filename in os.listdir(FETCH_PATH)
                if "x_val" in filename and "data1516" in filename
            ]
            data1516_xval_weights = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "weights_val" in filename and "data1516" in filename
            ]
            data1516_xval_categories = [
                np.load(FETCH_PATH / filename)
                for filename in os.listdir(FETCH_PATH)
                if "categories_val" in filename and "data1516" in filename
            ]

            data1516_xtrain = np.concatenate((data1516_xtrains), axis=0)
            data1516_x_train_cats = np.concatenate((data1516_xtrain_categories), axis=0)
            data1516_x_train_weights = np.concatenate((data1516_xtrain_weights), axis=0)

            data1516_xval = np.concatenate((data1516_xvals), axis=0)
            data1516_x_val_cats = np.concatenate((data1516_xval_categories), axis=0)
            data1516_x_val_weights = np.concatenate((data1516_xval_weights), axis=0)

            data1516_x_weights = np.concatenate(
                (data1516_x_train_weights, data1516_x_val_weights), axis=0
            )
            data1516_x_cats = np.concatenate(
                (data1516_x_train_cats, data1516_x_val_cats), axis=0
            )
            data1516 = np.concatenate((data1516_xtrain, data1516_xval), axis=0)
            
            
            print("Data 1516 mix partitioning and concatenation done")
            
            # * SM MC byte size

            print(" ")
            print(f"xtrain: {(xtrain.nbytes)/1000000000} Gbytes")
            print(f"xval: {(xval.nbytes)/1000000000} Gbytes")
            print(" ")

            # * Scaling
            self.column_trans = scaler
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            with strategy.scope():

                xtrain = self.column_trans.fit_transform(xtrain)
                xval = self.column_trans.transform(xval)

                data1516_xtrain = self.column_trans.transform(data1516_xtrain)
                data1516_xval = self.column_trans.transform(data1516_xval)
                data15_and_16_xtrain = self.column_trans.transform(data15_and_16_xtrain)
                data15_and_16_xval = self.column_trans.transform(data15_and_16_xval)

            # * SM background
            np.save(MERGE_PATH / f"Merged{megaset}_xtrain", xtrain)
            np.save(MERGE_PATH / f"Merged{megaset}_weights_train", x_train_weights)
            np.save(MERGE_PATH / f"Merged{megaset}_categories_train", x_train_cats)
            np.save(MERGE_PATH / f"Merged{megaset}_xval", xval)
            np.save(MERGE_PATH / f"Merged{megaset}_weights_val", x_val_weights)
            np.save(MERGE_PATH / f"Merged{megaset}_categories_val", x_val_cats)
            np.save(MERGE_PATH / f"Merged{megaset}_etmiss_val", xval_etmiss)
            
            print(f"SM MC megaset {megaset} saved")

            # * Data 15 and 16
            np.save(MERGE_PATH / f"Merged{megaset}_data15_and_16", data15_and_16)
            np.save(
                MERGE_PATH / f"Merged{megaset}_data15_and_16_weights",
                data15_and_16_x_weights,
            )
            np.save(
                MERGE_PATH / f"Merged{megaset}_data15_and_16_categories",
                data15_and_16_x_cats,
            )
            np.save(
                MERGE_PATH / f"Merged{megaset}_data15_and_16_etmiss",
                data15_and_16_etmiss,
            )
            
            print(f"Data 15 and data 16 megaset {megaset} saved")

            # * Data1516 mix
            np.save(MERGE_PATH / f"Merged{megaset}_data1516", data1516)
            np.save(
                MERGE_PATH / f"Merged{megaset}_data1516_weights", data1516_x_weights
            )
            np.save(
                MERGE_PATH / f"Merged{megaset}_data1516_categories", data1516_x_cats
            )
            np.save(MERGE_PATH / f"Merged{megaset}_data1516_etmiss", data1516_etmiss)
            np.save(MERGE_PATH / f"Merged{megaset}_data1516_isBSM", data1516_isBSM)
            
            print(f"Data 1516 mix megaset {megaset} saved")

        print("Megabatching done")
        print(" ")


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    global data_shape
    data_shape = 529

    L2 = DataHandling(DATA_PATH, True, SAVE_VAR, LOAD_VAR, lep=2, convert=True)
    #L2.convertParquet()
    #L2.createMCSubsamples()
    L2.mergeMegaBatches()