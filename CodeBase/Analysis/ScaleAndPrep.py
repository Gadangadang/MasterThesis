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

class ScaleAndPrep:
    def __init__(self, path: Path, event_rmm=False, save=False, load=False) -> None:
        """_summary_

        Args:
            path (str): _description_
            event_rmm (bool, optional):
        """
        self.path = path
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

        files = [
            f
            for f in listdir(self.path)
            if isfile(join(self.path, f))
            and f[-4:] != ".npy"
            and f[-4:] != ".csv"
            and f[-5:] != "_b.h5"
            and f[-4:] != ".txt"
            and f[-3:] != ".h5"
        ]
        
        
      
        return files  # type: ignore

    def fetchDfs(self) -> None:  # exlude=["data18", "ttbar"]
        """
        This function takes all dataframes stored as hdf5 files and adds them to a list,
        where this list later is used for scaling, splitting and merging of dataframes.

        """
        files = self.onlyfiles.copy()  # type: ignore
      
        self.dfs = []
        self.datas = []
        self.signals = []

        data_names = ["data15", "data16", "data17", "data18"]
        """signal_names = [
            "LRSMWR2400NR50",
            "LRSMWR4500NR400",
            "WeHNL5040Glt01ddlepfiltch1",
            "WeHNL5060Glt01ddlepfiltch1",
            "WeHNL5070Glt01ddlepfiltch1",
            "WmuHNL5040Glt01ddlepfiltch1",
            "WmuHNL5060Glt01ddlepfiltch1",
            "WmuHNL5070Glt01ddlepfiltch1",
            "ttbarHNLfullLepMLm15",
            "ttbarHNLfullLepMLm75",
            "ttbarHNLfullLepMLp15",
            "ttbarHNLfullLepMLp75",
        ]
        """
        
        signal_names = ["MGPy8EGA14N23LOC1N2WZ800p0p050p0p03L2L7", "MGPy8EGA14N23LOC1N2WZ450p0p0300p0p03L2L7"]
        charge_df_data = []

        self.tot_mc_events = 0
        self.tot_data_events = 0
        self.tot_signal_events = 0

        for file in files:

            df = pd.read_hdf(self.path / file)
            

            name = file[: file.find("_3lep")]
            print(df.columns, len(df.columns))
        

            if name in data_names:
                name = "data"
                
            if name in ["LRSMWR2400NR50",
            "LRSMWR4500NR400",
            "WeHNL5040Glt01ddlepfiltch1",
            "WeHNL5060Glt01ddlepfiltch1",
            "WeHNL5070Glt01ddlepfiltch1",
            "WmuHNL5040Glt01ddlepfiltch1",
            "WmuHNL5060Glt01ddlepfiltch1",
            "WmuHNL5070Glt01ddlepfiltch1",
            "ttbarHNLfullLepMLm15",
            "ttbarHNLfullLepMLm75",
            "ttbarHNLfullLepMLp15",
            "ttbarHNLfullLepMLp75",]:
                continue
            
            try:

                df.drop(
                    [
                        "nlep_BL",
                        "nlep_SG",
                        "ele_0_charge",
                        "ele_1_charge",
                        "ele_2_charge",
                        "muo_0_charge",
                        "muo_1_charge",
                        "muo_2_charge",
                    ],
                    axis=1,
                    inplace=True,
                )
            except:
                pass

            count = len(df)
            names = [name] * count
            names = np.asarray(names)

            df["Category"] = names

            if name == "data":
                self.datas.append(df)
                self.tot_data_events += len(df)
            elif name in signal_names:
                self.signals.append(df)
                self.tot_signal_events += len(df)
            else:
                self.dfs.append(df)
                self.tot_mc_events += len(df)

    def MergeScaleAndSplit(self):
        """_summary_"""

        if not self.load:
            plotRMMMatrix = plotRMM(self.path, rmm_structure, RMMSIZE)

            try:
                self.df  # type: ignore
            except:
                self.fetchDfs()

            df_train = []
            df_val = []

            df_train_w = []
            df_val_w = []

            df_train_cat = []
            df_val_cat = []

            for df in self.dfs:
                
            
                print(df["Category"].unique())
      
                weight = df["wgt_SG"]
                
                
                

                print(np.sum(weight))
                flag = 1
                while flag:
                    x_b_train, x_b_val = train_test_split(
                        df, test_size=0.2, random_state=seed
                    )

                    weights_train = x_b_train["wgt_SG"]
                    weights_val = x_b_val["wgt_SG"]

                    ratio = np.sum(weights_train) / np.sum(weight) * 100

                    if ratio < 81 and ratio > 79:
                        print(np.sum(weights_train))
                        print(f"Ratio: {ratio:.2f}%")
                        break

                train_categories = x_b_train["Category"]
                val_categories = x_b_val["Category"]

                df_train.append(x_b_train)
                df_train_w.append(weights_train)
                df_val.append(x_b_val)
                df_val_w.append(weights_val)

                df_train_cat.append(train_categories)
                df_val_cat.append(val_categories)

            X_b_train = pd.concat(df_train)
            X_b_val = pd.concat(df_val)
            
            all_cols = X_b_train.columns
            cols = []
            for col in all_cols:
                if col == "Category":
                    continue
                try:
                    
                    print(col, np.mean(X_b_train[col]), np.max(X_b_train[col]), np.min(X_b_train[col]))
                    if np.min(X_b_train[col]) < -3 or np.max(X_b_train[col]) > 3:
                        cols.append(col)
                        print(f"{col} added to be scaled later ")
                except:
                    continue
            
            print("Col to be scaled found")


            print(" ")
            print("Concatination started")
            print(" ")
            self.train_categories = pd.concat(df_train_cat)
            self.val_categories = pd.concat(df_val_cat)

            self.weights_train = pd.concat(df_train_w)
            self.weights_val = pd.concat(df_val_w)

            self.data = pd.concat(self.datas)
            self.signal = pd.concat(self.signals)
            
            print(" ")
            print("Concatination done for dfs")
            print(" ")

            self.data_categories = self.data["Category"]
            self.data_weights = self.data["wgt_SG"]
            
            self.signal_categories = self.signal["Category"]
            self.signal_weights = self.signal["wgt_SG"]
            
            self.X_train_trilep_mass = X_b_train["TrileptonMass"]
            self.X_val_trilep_mass = X_b_val["TrileptonMass"]
            self.data_trilep_mass = self.data["TrileptonMass"]
            self.signal_trilep_mass = self.signal["TrileptonMass"]
            

            channels = [
                "Zeejets",
                "Zmmjets",
                "Zttjets",
                "diboson2L",
                "diboson3L",
                "diboson4L",
                "higgs",
                "singletop",
                "topOther",
                "Wjets",
                "triboson",
                "ttbar",
            ]
            
            print(" ")
            print("Random event sampling")
            print(" ")
            # Indentifying Zeejets events for rmm single event plotting
            idx_rmm = []
            choices = random.sample(channels, 4)
            for choice in choices:

                id_rmm = np.where(self.train_categories == choice)[0]

                idx_rmm.append((choice, id_rmm))

            self.idxs = []

            for channel in channels:

                idx_val = np.where(X_b_val["Category"] == channel)[0]
                idx_train = np.where(X_b_train["Category"] == channel)[0]
                id = (channel, idx_train, idx_val)

                self.idxs.append(id)

            X_b_train.drop("Category", axis=1, inplace=True)
            X_b_val.drop("Category", axis=1, inplace=True)
            self.data.drop("Category", axis=1, inplace=True)
            self.signal.drop("Category", axis=1, inplace=True)

            X_b_train.drop("wgt_SG", axis=1, inplace=True)
            X_b_val.drop("wgt_SG", axis=1, inplace=True)
            self.data.drop("wgt_SG", axis=1, inplace=True)
            self.signal.drop("wgt_SG", axis=1, inplace=True)
            
            X_b_train.drop("TrileptonMass", axis=1, inplace=True)
            X_b_val.drop("TrileptonMass", axis=1, inplace=True)
            self.data.drop("TrileptonMass", axis=1, inplace=True)
            self.signal.drop("TrileptonMass", axis=1, inplace=True)
            
            X_b_train["flcomp"].to_hdf(DATA_PATH / "flcomp_train.h5", "mini")
            X_b_val["flcomp"].to_hdf(DATA_PATH / "flcomp_val.h5", "mini")
            self.data["flcomp"].to_hdf(DATA_PATH / "flcomp_data.h5", "mini")
            self.signal["flcomp"].to_hdf(DATA_PATH / "flcomp_data.h5", "mini")
            
            self.flcomp_train = X_b_train["flcomp"]
            self.flcomp_val = X_b_val["flcomp"]           
            self.flcomp_data = self.data["flcomp"]      
            self.flcomp_signals = self.signal["flcomp"]   
            
            X_b_train.drop("flcomp", axis=1, inplace=True)
            X_b_val.drop("flcomp", axis=1, inplace=True)
            self.data.drop("flcomp", axis=1, inplace=True)
            self.signal.drop("flcomp", axis=1, inplace=True)

            # print(X_b_train.columns)

            print(" ")
            print(np.shape(X_b_val))
            print(" ")
    
                
            print(" ")
            print("Scaling initiated ... ")
            print(" ")
            
            
            

            if scaler == "MinMax":
            
                column_trans = ColumnTransformer(
                        [('scaler_ae', scaler, cols)],
                        remainder='passthrough'
                    )
            else:
                column_trans = scaler
            
            #column_trans = scaler
            
            self.noscale_X_b_train = X_b_train.copy().to_numpy()

            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            with strategy.scope():

                self.X_b_train = column_trans.fit_transform(X_b_train)
                self.X_b_val = column_trans.transform(X_b_val)
                self.data = column_trans.transform(self.data)
                self.signal = column_trans.transform(self.signal)
                
            print(" ")
            print("Scaling done")
            print(" ")
            
            print(" ")
            print(np.shape(self.X_b_train))
            print(" ")

            if self.event_rmm:
                for choice, idxss in idx_rmm:
                    event = random.sample(list(idxss), 1)[0]
                    # print(event, choice)
                    plotRMMMatrix.plotDfRmmMatrixNoMean(self.X_b_train, choice, event)

            ### Plot RMM for each channel

            for channel, idx_train, idx_val in self.idxs:
                cols = X_b_train.columns
                train = self.X_b_train[idx_train]
                val = self.X_b_val[idx_val]
                train = pd.DataFrame(data=train, columns=cols)
                val = pd.DataFrame(data=val, columns=cols)
                plot_df = pd.concat([train, val])
                plotRMMMatrix.plotDfRmmMatrix(plot_df, channel)

            datapoint_string = f"""
    Total MC events: {self.tot_mc_events} \n
    Total Data events: {self.tot_data_events} \n
    Total Signal events: {self.tot_signal_events} \n
                """
            print(datapoint_string)
            
            self.columns = np.asarray(X_b_train.columns, dtype=str)
            
            self.scalecols = np.asarray(cols, dtype=str)
            

            if self.save:
                np.save(DATA_PATH / "X_train_small.npy", self.X_b_train[::1000])
                np.save(DATA_PATH / "X_train.npy", self.X_b_train)
                np.save(DATA_PATH / "X_val.npy", self.X_b_val)
                np.save(DATA_PATH / "Data.npy", self.data)
                np.save(DATA_PATH / "signal.npy", self.signal)
                np.save(DATA_PATH / "cols.npy", self.columns)
                np.save(DATA_PATH / "scalecols.npy", self.scalecols)

                np.save(DATA_PATH / "channel_names.npy", np.asarray(channels))
                for row in self.idxs:
                    name = row[0]
                    np.save(DATA_PATH / f"{name}_train_idxs.npy", row[1])
                    np.save(DATA_PATH / f"{name}_val_idxs.npy", row[2])

                # Using _b to separate these hdf5 files from the sample files
                X_b_train.to_hdf(DATA_PATH / "X_b_train.h5", "mini")
                self.train_categories.to_hdf(DATA_PATH / "train_cat_b.h5", "mini")
                self.val_categories.to_hdf(DATA_PATH / "val_cat_b.h5", "mini")

                self.weights_train.to_hdf(DATA_PATH / "train_weight_b.h5", "mini")
                self.weights_val.to_hdf(DATA_PATH / "val_weight_b.h5", "mini")

                self.data_categories.to_hdf(DATA_PATH / "data_cat_b.h5", "mini")
                self.data_weights.to_hdf(DATA_PATH / "data_weight_b.h5", "mini")
                
                self.signal_categories.to_hdf(DATA_PATH / "signal_cat_b.h5", "mini")
                self.signal_weights.to_hdf(DATA_PATH / "signal_weight_b.h5", "mini")
                
            
                self.X_train_trilep_mass.to_hdf(DATA_PATH / "X_train_trilep.h5", "mini")
                self.X_val_trilep_mass.to_hdf(DATA_PATH / "X_val_trilep.h5", "mini")
                self.data_trilep_mass.to_hdf(DATA_PATH / "data_trilep.h5", "mini")
                self.signal_trilep_mass.to_hdf(DATA_PATH / "signal_trilep.h5", "mini")
                
                #self.columns.to_hdf(DATA_PATH / "cols.h5", "mini")

        else:
            self.X_b_train_small = np.load(DATA_PATH / "X_train_small.npy")
            self.X_b_train = np.load(DATA_PATH / "X_train.npy")
            self.X_b_val = np.load(DATA_PATH / "X_val.npy")
            self.data = np.load(DATA_PATH / "Data.npy")
            self.scalecols = np.load(DATA_PATH / "scalecols.npy")
            self.signal = np.load(DATA_PATH / "signal.npy")
            
            
          
            self.cols = np.load(DATA_PATH / "cols.npy")#pd.read_hdf(DATA_PATH / "cols.h5")
            
            
            self.idxs = []
            channels = np.load(DATA_PATH / "channel_names.npy")
            for name in channels:

                train = np.load(DATA_PATH / f"{name}_train_idxs.npy")
                val = np.load(DATA_PATH / f"{name}_val_idxs.npy")
                self.idxs.append((name, train, val))

            # print(self.idxs)

            # Using _b to separate these hdf5 files from the sample files
            self.train_categories = pd.read_hdf(DATA_PATH / "train_cat_b.h5")
            self.val_categories = pd.read_hdf(DATA_PATH / "val_cat_b.h5")

            self.weights_train = pd.read_hdf(DATA_PATH / "train_weight_b.h5")
            self.weights_val = pd.read_hdf(DATA_PATH / "val_weight_b.h5")

            self.data_categories = pd.read_hdf(DATA_PATH / "data_cat_b.h5")
            self.data_weights = pd.read_hdf(DATA_PATH / "data_weight_b.h5")
            
            self.signal_categories = pd.read_hdf(DATA_PATH / "signal_cat_b.h5")
            self.signal_weights = pd.read_hdf(DATA_PATH / "signal_weight_b.h5")
            
            
            self.flcomp_train = pd.read_hdf(DATA_PATH / "flcomp_train.h5")
            self.flcomp_val = pd.read_hdf(DATA_PATH / "flcomp_val.h5")
            self.flcomp_data = pd.read_hdf(DATA_PATH / "flcomp_data.h5")
            self.noscale_X_b_train = pd.read_hdf(DATA_PATH / "X_b_train.h5")
            
            self.X_train_trilep_mass = pd.read_hdf(DATA_PATH / "X_train_trilep.h5")
            self.X_val_trilep_mass = pd.read_hdf(DATA_PATH / "X_val_trilep.h5")
            self.data_trilep_mass = pd.read_hdf(DATA_PATH / "data_trilep.h5")
            self.signal_trilep_mass = pd.read_hdf(DATA_PATH / "signal_trilep.h5")
            
            