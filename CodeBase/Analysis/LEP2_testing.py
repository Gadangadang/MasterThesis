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
    def __init__(self, path: Path, event_rmm=False, save=False, load=False, lep=3, convert=True) -> None:
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
        self.convert = convert
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
                
                
                
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
                with strategy.scope():
                    start = file.find("two_")
                    stop = file.find("_3lep")
                    name = file[start+4:stop]
                    if name in ["data15", "data16", "data17","data18","singletop","Diboson","Zeejets1","Zeejets2","Zeejets3","Zmmjets1","Zmmjets2","Zmmjets3""Zttjets","Wjets","ttbar","Zeejets4"]:
                        continue
                    
                    df = pd.read_hdf(self.path/file)
                    print(df.columns)
                    break
                    #scaled_df = scaler.fit_transform(df)
                    name = "twolep_" + name +".parquet"
                    
                    df.to_parquet(self.path/name)
                    print(f"{name} done")
                
            

            
        
                
        self.parqs = [f for f in listdir(self.path) if isfile(join(self.path, f)) and f[-8:] == ".parquet"]
        
        print(self.parqs)

    def createMCSubsamples(self):
        """
        Create subsets that maintains the SM MC distribution. 
        
        
        
        
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            with strategy.scope():
                df = pl.read_parquet(self.path/file, use_pyarrow=True)
                
                
            print("Scaling done")
        
        """
        
        try:
            self.parqs
        except:
            self.convertParquet()
        
        
        
        
        for file in self.parqs:
            if file[7:11] == "data":
                continue
            
            start = file.find("twolep_")
            end = file.find(".parquet")
            
            name = file[start+7:end]
            
            print(name)
            

            df = pl.read_parquet(self.path/file, use_pyarrow=True)
            
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
            
            x_b_train, x_b_val = train_test_split(
                        df, test_size=0.2, random_state=seed
            )
            
            self.sampleSet(x_b_train, x_b_val, name)
            
            break
            
            
            
            

        
        
    def sampleSet(self, xtrain, xval, name):
        
        """count = len(df)
        names = [name] * count
        names = np.asarray(names)

        df["Category"] = names"""
        
        print(xtrain.columns)
        
        print(np.shape(xtrain))
        
        
        
        
        #* Sample from training and validation set
        
        indices_train = np.asarray(range(len(xtrain)))
        
        np.random.shuffle(indices_train)
    
        split_idx_train = np.array_split(indices_train, 10)
        
        
        indices_val = np.asarray(range(len(xval)))
        
        np.random.shuffle(indices_val)
        
        split_idx_val = np.array_split(indices_val, 10)
        
        megaset = 0
        for idx_set_train, idx_set_val in zip(split_idx_train, split_idx_val):
            
            
            
            weights_train = xtrain["wgt_SG"].to_numpy()[idx_set_train]
            weights_val = xval["wgt_SG"].to_numpy()[idx_set_val]
            
            #train_categories = xtrain["Category"].to_numpy()[idx_set_train]
            #val_categories = xval["Category"].to_numpy()[idx_set_val]
            
            
            #xtrain = xtrain.drop("__index_level_0__")
            #xval = xval.drop("__index_level_0__")
            #xtrain = xtrain.drop("Category")
            #xtrain = xtrain.drop("wgt_SG")
            
            #xval = xval.drop("Category")
            #xval = xval.drop("wgt_SG")

            np.save(DATA_PATH/ f"Megabatches/MSET{megaset}_{name}_weights_train", weights_train)
            np.save(DATA_PATH/ f"Megabatches/MSET{megaset}_{name}_weights_val", weights_val)
            
            #np.save(DATA_PATH/ f"Megasbatches/MSET{megaset}_{name}_categories_train", train_categories)
            #np.save(DATA_PATH/ f"Megasbatches/MSET{megaset}_{name}_categories_val", val_categories)
            
            megaset += 1
            
            
        
        
    
        
        
        
if __name__ == "__main__":
    L2 = LEP2ScaleAndPrep(DATA_PATH, True, SAVE_VAR, LOAD_VAR, lep=2, convert=True)
    L2.convertParquet()
    L2.createMCSubsamples()