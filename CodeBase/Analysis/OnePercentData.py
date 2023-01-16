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
    useclass = RunVAE
elif TYPE == "AE":
    useclass = RunAE
    
class OnePercentData(useclass):
    def __init__(self, data_structure:object, path: str)->None:
        super().__init__(data_structure, path)
        
    def run(self)->None:
        """_summary_
        """
        st = time.time()
        
        #* Fetch events from the MC set
        
        self.tot_set_train_idxs_list = []
        self.tot_set_val_idxs_list = []
        
        for channel, idx_train, idx_val in self.idxs:
            print(f"Channel: {channel}")
            
            nr_channel_events_train = len(idx_train)
            one_percent_no_events_t = int(nr_channel_events_train/100)
            new_indices_train = np.random.choice(idx_train, size=one_percent_no_events_t, replace=False)  
            
            weights = self.err_train[idx_val].copy()
            
            one_percent_weights = self.err_train[new_indices_train].copy()
            
            #* Fetch randomly selected events for training
            flag = True
            while flag:
                nr_channel_events_train = len(idx_train)
                one_percent_no_events_t = one_percent_no_events_t + 15
                print(one_percent_no_events_t)
                new_indices_train = np.random.choice(idx_train, size=one_percent_no_events_t, replace=False)  
                
                weights = self.err_train[idx_train].copy()
                
                one_percent_weights = self.err_train[new_indices_train].copy()
                
                
                
                if np.abs(np.sum(one_percent_weights) - (np.sum(weights)/100)) > 100:
                    one_percent_no_events_t = int(nr_channel_events_train/100)
                if np.abs(np.sum(one_percent_weights) - (np.sum(weights)/100)) < 1:
                    flag = False
            
            self.tot_set_train_idxs_list.append(new_indices_train)
                        
            nr_channel_events_val= len(idx_val)
            one_percent_no_events_v = int(nr_channel_events_val/100)
            weights_val = self.err_val[idx_val].copy()
            
            new_indices_val = np.random.choice(idx_val, size=one_percent_no_events_v, replace=False) 
            
            one_percent_weights = self.err_train[new_indices_val].copy()
            
            #* Fetch randomly selected events for validation
            flag2 = True
            while flag2:
                nr_channel_events_val = len(idx_val)
                one_percent_no_events_v = one_percent_no_events_v + 1
                new_indices_val = np.random.choice(idx_val, size=one_percent_no_events_v, replace=False)  
                
                weights_val = self.err_val[idx_val].copy()
                
                one_percent_weights = self.err_val[new_indices_val].copy()
                
                
                if np.abs(np.sum(one_percent_weights) - (np.sum(weights_val)/100)) > 50:
                    one_percent_no_events_v = int(nr_channel_events_val/100)
                    
                if np.abs(np.sum(one_percent_weights) - (np.sum(weights_val)/100)) < 1:
                    flag2 = False
            
            self.tot_set_val_idxs_list.append(new_indices_val)
            
        print(f"Number of selected events: {one_percent_no_events_t + one_percent_no_events_v}")    
            
        self.tot_set_train_idxs = np.concatenate(self.tot_set_train_idxs_list, axis=0)
        self.tot_set_val_idxs = np.concatenate(self.tot_set_val_idxs_list, axis=0)
        
        self.tot_weights_per_channel = []
        self.tot_data = []
        
        self.act_weights = []
        start = 0
        
        for id, idx_train  in enumerate(self.tot_set_train_idxs_list):
            idx_val = self.tot_set_val_idxs_list[id]
            
            x_tot = self.X_val[idx_val]
            self.tot_data.append(x_tot)
            
            act = self.data_structure.weights_val.to_numpy().copy()[idx_val]
            
            self.act_weights.append(act)
            idxs = np.concatenate((idx_train, idx_val), axis=0)
            
            end = start + len(idx_val)
            
            print(len(x_tot), len(idx_val))
            
            self.tot_weights_per_channel.append(np.asarray(range(start, end)))
            
            start = end - 1
            
        
        X_tot = np.concatenate(self.tot_data, axis=0)
        self.act_weights = np.concatenate(self.act_weights, axis=0)
        
        X_train = self.X_train[self.tot_set_train_idxs]
        X_val = self.X_val[self.tot_set_val_idxs]
        
        sample_weight_n = self.data_structure.weights_train.to_numpy().copy()[
            self.tot_set_train_idxs
        ]
        
        sample_weight = pd.DataFrame(sample_weight_n)

        self.err_val = self.act_weights
        
        
        #* Fetch events from the data
        nr_data_events = len(self.data) 
        indices = range(nr_data_events)
        one_percent_no_events = int(nr_data_events/100)
        new_indices = np.random.choice(indices, size=one_percent_no_events, replace=False)        
        
        self.dummysample_dataset = self.data[new_indices]
        self.sig_err = np.ones(len(self.dummysample_dataset))
        
        #* Check weight comparison
        print(" ")
        print("*****************************************")
        print(f"Data weights: {np.sum(self.sig_err):.1f} | MC weights: {np.sum(self.err_val):.1f}")
        print("*****************************************")
        print(" ")
        
        
        #* Tuning, training, and inference
        """HPT = HyperParameterTuning(self.data_structure, STORE_IMG_PATH)
        HPT.runHpSearch(
            X_train, X_val, sample_weight, small=SMALL
        )
        
        self.AE_model = HPT.AE_model
        """
        

        self.trainModel(X_train, X_val, sample_weight)

        
        self.runInference(X_tot, self.dummysample_dataset, True)

        self.checkReconError(self.channels, sig_name="1%_ATLAS_Data")
        
        et = time.time()
       
        
        img_path = Path(f"histo/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_1%_ATLAS_Data.pdf")
        path = STORE_IMG_PATH/img_path

        files = {"photo":open(path, "rb")}
        message = f"Done calculating 1% data plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
        resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
        print(resp.status_code)
        
        
           
