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


class DummyData(model):
    def __init__(self, data_structure:object, path:str)->None:
        super().__init__(data_structure, path)
        
    def swapEventsInChannels(self, fraction_cols:float, fraction_swap:float)->None:
        """_summary_

        Args:
            fraction_cols (float): _description_
            fraction_swap (float): _description_
        """
        st = time.time()
        
        rows, cols = np.shape(self.X_val)
        
        nr_rows_swap = int(rows*fraction_swap)
        
        nr_cols_swap = int(cols*fraction_cols)
        
        cols_to_swap = np.random.choice(range(cols), size=nr_cols_swap, replace=False)
        
        rows_to_swap = np.random.choice(range(rows), size=nr_rows_swap, replace=False)
        
        old_row = rows_to_swap.copy()
        
        np.random.shuffle(rows_to_swap)
        
        pairs = []
        
        train_cat = self.data_structure.train_categories.to_numpy()
        val_cat = self.data_structure.val_categories.to_numpy()
        
        for i in range(0, len(rows_to_swap), 2):
            
            col = np.random.choice(cols_to_swap, size=1, replace=False)[0]
            #print(col)
            pairs.append((col, rows_to_swap[i], rows_to_swap[i+1]))   
        
        X_val_dummy = self.X_val.copy()
        
        for column_number in cols_to_swap:
            
            np.random.shuffle(rows_to_swap)
        
            X_val_dummy[old_row, column_number] = X_val_dummy[rows_to_swap, column_number]
            
        val_cat[old_row] = "Signal" 
            
        

            #print(X_val_dummy[row_1, column_number], X_val_dummy[row_2, column_number])
        
        #val_cat = np.concatenate((train_cat, val_cat), axis=0)
      
        
        #X_tot = np.concatenate((self.X_train, X_val_dummy), axis=0)
        
      
        
        
        signal = X_val_dummy[np.where(val_cat == "Signal")]
        X_tot = X_val_dummy[np.where(val_cat != "Signal")]
        
        
        
        sample_weight_t = self.data_structure.weights_train.to_numpy().copy()
        sample_weight_v = self.data_structure.weights_val.to_numpy().copy()
        
        sample_weight = pd.DataFrame(sample_weight_t)
        
        self.err_val = sample_weight_v#np.concatenate((sample_weight_t, sample_weight_v), axis=0)
        
        self.sig_err = self.err_val[np.where(val_cat == "Signal")]
        self.err_val = self.err_val[np.where(val_cat != "Signal")]
        
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

        
       
        self.checkReconError(self.channels, sig_name="Dummydata")   
        
        et = time.time()
        
        
        img_path = Path(f"histo/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_Dummydata.pdf")
        path = STORE_IMG_PATH/img_path

        files = {"photo":open(path, "rb")}
        message = f"Done calculating dummy data plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
        resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
        print(resp.status_code)
                    
 