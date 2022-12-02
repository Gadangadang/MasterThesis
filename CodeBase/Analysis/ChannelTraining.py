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
    
    
class ChannelTraining(RunAE):
    def __init__(self,data_structure:object, path:str)->None:
        super().__init__(data_structure, path)
        
    def run(self, small=SMALL)->None:
        """_summary_

        Args:
            small (bool, optional): Choice to use big or small network, True -> small network. 
                                    Defaults to False.
        """
        
        
        #self.data_structure.weights_val = self.data_structure.weights_val.to_numpy()

        for channel, idx_train, idx_val in self.idxs:
            
            if channel in ["Zeejets", "Zmmjets", "Zttjets", "diboson2L", "diboson3L"]:
                continue

            st = time.time()
            channels = self.channels.copy()
            channels.remove(channel)

            print(f"Channel: {channel}  started")

            new_index = np.delete(np.asarray(range(len(self.X_train))), idx_train)
            new_index_val = np.delete(np.asarray(range(len(self.X_val))), idx_val)

            self.val_cats = self.data_structure.val_categories.to_numpy().copy()[
                new_index_val
            ]

            self.err_val = self.data_structure.weights_val.to_numpy().copy()[new_index_val]

            X_train_reduced = self.X_train.copy()[new_index]
            X_val_reduced = self.X_val.copy()[new_index_val]

            sample_weight = self.data_structure.weights_train.to_numpy().copy()[
                new_index
            ]
            sample_weight = pd.DataFrame(sample_weight)

            channel_train_set = self.X_train.copy()[idx_train]
            channel_val_set = self.X_val.copy()[idx_val]

            signal = channel_val_set  # np.concatenate((channel_train_set, channel_val_set), axis=0)

            sig_err_t = self.data_structure.weights_train.to_numpy()[
                np.where(self.data_structure.train_categories == channel)[0]
            ]

            sig_err_v = self.data_structure.weights_val.to_numpy()[
                np.where(self.data_structure.val_categories == channel)[0]
            ]

            self.sig_err = sig_err_v  # np.concatenate((sig_err_t, sig_err_v), axis=0)

            self.name = "no_" + channel

            HPT = HyperParameterTuning(self.data_structure, STORE_IMG_PATH)
            HPT.runHpSearch(
                X_train_reduced, X_val_reduced, sample_weight, small=small, epochs=2
            )

            # self.trainModel(X_train_reduced, X_val_reduced, sample_weight)
            print(" ")
            print("Hyperparam search done")
            print(" ")

            self.AE_model = HPT.AE_model

            
            self.trainModel(X_train_reduced, X_val_reduced, sample_weight)

            #print(self.AE_model.layers[1].weights)
            self.runInference(X_val_reduced, signal, True)

            self.checkReconError(channels, sig_name=channel)
            
            et = time.time()
            

            img_path = Path(f"histo/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{channel}.pdf")
            path = STORE_IMG_PATH/img_path

            files = {"photo":open(path, "rb")}
            message = f"Done calculating sig: {channel} plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
            resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
            print(resp.status_code)

