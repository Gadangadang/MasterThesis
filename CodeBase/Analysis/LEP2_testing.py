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
                
                break
                
                
                break
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
                with strategy.scope():
                    start = file.find("two_")
                    stop = file.find("_3lep")
                    name = file[start+4:stop]
                    if name in ["data15", "data16", "data17","data18","singletop","Diboson","Zeejets1","Zeejets2","Zeejets3","Zmmjets1","Zmmjets2","Zmmjets3""Zttjets","Wjets","ttbar","Zeejets4"]:
                        continue
                    
                    df = pd.read_hdf(self.path/file)
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
            
            df = pl.read_parquet(self.path/file, use_pyarrow=True)
            
            x_b_train, x_b_val = train_test_split(
                        df, test_size=0.2, random_state=seed
            )
            
            self.sampleSet(x_b_train, x_b_val)
            
            
            
            

        
        
    def sampleSet(self, xtrain, xval):
        
        print(len(xtrain))
        percentage = int(len(xtrain)/10)
        print(percentage)
        
        
        #* Sample from training set
        indices = np.asarray(range(len(xtrain)))
        
        np.random.shuffle(indices)
    
        split_idx = np.array_split(indices, percentage)
        print(split_idx)
        
        exit()
        
        lenght_train = len(xtrain)
        
        weights_train = xtrain["wgt_SG"]
        weights_val = xval["wgt_SG"]
        
        train_categories = xtrain["Category"]
        val_categories = xval["Category"]
        
        #* Sample from validation set
        
        np.random.shuffle(arr)
    
        split_idx = np.array_split(arr, 3)
        
        
            
            

if __name__ == "__main__":
    L2 = LEP2ScaleAndPrep(DATA_PATH, True, SAVE_VAR, LOAD_VAR, lep=2, convert=True)
    L2.convertParquet()
    L2.createMCSubsamples()