import time
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
    
    
    
class SignalDumVeri(model):
    def __init__(self,data_structure:object, path:str)->None:
        super().__init__(data_structure, path)
        
    
    def run(self):
        st = time.time()
        
        self.signal_cats = self.signal_cats.to_numpy()
        
        print(np.unique(self.signal_cats))
        #breakpoint()
        i = 0
        for signal in np.unique(self.signal_cats):
            
            signal_name = signal
            """if i == 0:
                i+=1
                continue"""

            
            val_cat = self.data_structure.val_categories
            sample_weight = self.data_structure.weights_train
            
            
            self.sig_err = self.signal_weights.to_numpy()[np.where(self.signal_cats == signal_name )]
            signal = self.signal[np.where(self.signal_cats == signal_name )]
            
            
            
            #* Tuning, training, and inference
            if TYPE == "AE":
                if i == 0:
                    HPT = HyperParameterTuning(self.data_structure, STORE_IMG_PATH)
                    HPT.runHpSearch(
                        self.X_train, self.X_val, sample_weight, small=SMALL, epochs=3
                    )
                    self.AE_model = HPT.AE_model
                    i+=1
                

            self.trainModel(self.X_train, self.X_val, sample_weight)

            
            self.runInference(self.X_val, signal, True)
            
        
            self.checkReconError(self.channels, sig_name=f"{signal_name[21:31]}")   
            
            et = time.time()
            
            
            img_path = Path(f"histo/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{signal_name[21:31]}.pdf")
            path = STORE_IMG_PATH/img_path

            files = {"photo":open(path, "rb")}
            message = f"Done calculating dummy signal plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
            resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
            print(resp.status_code)
            