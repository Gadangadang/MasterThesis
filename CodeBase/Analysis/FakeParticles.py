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

from AE import RunAE
from VAE import RunVAE
from plotRMM import plotRMM
from Utilities.config import *
from Utilities.pathfile import *
from HyperParameterTuning import HyperParameterTuning

seed = tf.random.set_seed(1)

tf.keras.utils.get_custom_objects()["leaky_relu"] = tf.keras.layers.LeakyReLU()


scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}
scaler = scalers[SCALER]

if SMALL:
    arc = "small"
else:
    arc = "big"
    
        
if TYPE == "VAE":
    model = RunVAE
elif TYPE == "AE":
    model = RunAE


class FakeParticles(model):
    def __init__(self, data_structure:object, path:str)->None:
        super().__init__(data_structure, path)     
        
    
    def split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
        
    def run(self, cols_swap_keys:list, fraction_events: float) -> None:
        st = time.time()
        
        plotRMMMatrix = plotRMM(DATA_PATH, rmm_structure, 15)
        
        
        
        rows, cols = np.shape(self.X_val)
        
        nr_rows_swap = int(rows*fraction_events)
        
        print(nr_rows_swap)
        
        #nr_cols_swap = int(cols*fraction_cols)
        
        rows_to_swap = np.random.choice(range(rows), size=nr_rows_swap, replace=False)
        
        print(len(rows_to_swap))
        
        subset_partitioning = int(len(rows_to_swap)/len(cols_swap_keys))
        print(f"subset {subset_partitioning}")
        
        rows_split_up = list(self.split(rows_to_swap, len(cols_swap_keys)))
        rows_split_up = np.asarray(rows_split_up)
        
        
       
        
        print(len(rows_split_up), rows_split_up)
        
        val_cat = self.data_structure.val_categories.to_numpy()
        
        X_val_dummy = self.X_val.copy()
        
        event_list = np.random.choice(rows_split_up, size=1, replace=False)[0]
        event = np.random.choice(event_list, size=1, replace=False)[0]
            
        print(event)
        channel_test = val_cat[event]
        print(channel_test)
        plotRMMMatrix.plotDfRmmMatrixNoMean(X_val_dummy, channel_test, event, additional_info="preswap", fake=True)
        
        print(" ")
        print(" loop ")
        for idx, pair in enumerate(cols_swap_keys):
            rows = rows_split_up[idx]
            old_col = pair[0]
            new_col = pair[1]
            print(rows, old_col, new_col)
            
            
            for column in range(old_col, cols, 15):
                if column > cols:
                    break
                
                old_val = X_val_dummy[rows, column]
                new_val = X_val_dummy[rows, column + new_col-1]
                
                

                X_val_dummy[rows, column] = new_val
                X_val_dummy[rows, column + new_col-1] = old_val
                
                #print(X_val_dummy[rows, column], X_val_dummy[rows, column + new_col-1])
            
            val_cat[rows] = "Signal" 
            print(idx, len(rows))
        
        print(" loop end ")     
        print(" ")
          
            
        plotRMMMatrix.plotDfRmmMatrixNoMean(X_val_dummy, channel_test, event, fake=True)

        
        print("  ")
        print(np.where(val_cat == "Signal"))
        signal = X_val_dummy[np.where(val_cat == "Signal")]
        print(np.shape(signal))
        X_tot = X_val_dummy[np.where(val_cat != "Signal")]
        print(np.shape(X_tot))
        sample_weight_t = self.data_structure.weights_train.to_numpy().copy()
        sample_weight_v = self.data_structure.weights_val.to_numpy().copy()
        
        sample_weight = pd.DataFrame(sample_weight_t)
        
        self.err_val = sample_weight_v#np.concatenate((sample_weight_t, sample_weight_v), axis=0)
        
        self.sig_err = self.err_val[np.where(val_cat == "Signal")]
        self.err_val = self.err_val[np.where(val_cat != "Signal")]
        
        
        print(np.where(val_cat == "Signal"), np.shape(self.sig_err) , np.shape(self.err_val))
        
        #self.val_cats = np.concatenate((train_cat, val_cat), axis=0)
        self.val_cats = self.val_cats[np.where(val_cat != "Signal")]
        
        
        #* Tuning, training, and inference
        if TYPE == "AE":
            HPT = HyperParameterTuning(self.data_structure, STORE_IMG_PATH)
            HPT.runHpSearch(
                self.X_train, X_val_dummy, sample_weight, small=SMALL, epochs=3
            )
            self.AE_model = HPT.AE_model
        
        self.trainModel(self.X_train, X_val_dummy, sample_weight)
        self.runInference(X_tot, signal,True)
        self.checkReconError(self.channels, sig_name="FakeMC")     

        
        
        
        et = time.time()
            
        try:
            img_path = Path(f"histo/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_fakedata.pdf")
            path = STORE_IMG_PATH/img_path

            files = {"photo":open(path, "rb")}
            message = f"Done calculating dummy data plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
            resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
            print(resp.status_code)
        except:
            pass
            
        