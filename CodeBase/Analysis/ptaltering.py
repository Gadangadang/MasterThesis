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

class pTAltering(model):
    def __init__(self, data_structure:object, path:str)->None:
        super().__init__(data_structure, path)  
        
    def run(self, ptboost:list)->None:
        """_summary_

        Args:
            ptboost (list): List containing the number with which we scale the pT, thus the ET
        """
        
        
        X_val = self.X_val.copy()
        
        
        
        
        for index, ptscaling in enumerate(ptboost):
            X_val = self.X_val.copy()
            st = time.time()
            self.val_cats = self.data_structure.val_categories.to_numpy().copy()
            val_cat = self.data_structure.val_categories.to_numpy().copy()
            
            events = np.random.choice(range(np.shape(X_val)[0]), size=int(len(X_val)/100), replace=False)
            val_cat[events] = "Signal"
            
          
            
      
            
            #* Scale E_T
            signal = X_val[np.where(val_cat == "Signal")]
            X_val = X_val[np.where(val_cat != "Signal")]
            
            cols = np.load(DATA_PATH / "dfcols.npy", allow_pickle=True)
            
            pt_ele0 = np.where(cols == "e_T_ele_0")
            pt_muo0 = np.where(cols == "e_T_muo_0")
            
            df = pd.DataFrame(signal, columns = cols)
            indexes1 = df.index[df["e_T_ele_0"] > 0.].to_list()
            indexes2 = df.index[df["e_T_muo_0"] > 0.].to_list()
            
            delta_et_ele1ele2 = np.where(cols == "delta_e_t_ele_1")
            
            e1 = signal[indexes1, pt_ele0] * (signal[indexes1, delta_et_ele1ele2]-1)/(-(signal[indexes1, delta_et_ele1ele2]+1))
            signal[indexes1, pt_ele0] = signal[indexes1, pt_ele0]*ptscaling
            e0 = signal[indexes1, pt_ele0]
            signal[indexes1, delta_et_ele1ele2] = (e0-e1)/(e0+e1)
            
            delta_et_muo1 = np.where(cols == "delta_e_t_muo_1")
            
            e1 = signal[indexes2, pt_muo0] * (signal[indexes2, delta_et_muo1]-1)/(-(signal[indexes2, delta_et_muo1]+1))
            signal[indexes2, pt_muo0] = signal[indexes2, pt_muo0]*ptscaling
            e0 = signal[indexes2, pt_muo0]
            signal[indexes2, delta_et_muo1] = (e0-e1)/(e0+e1)
            
            print(np.shape(signal[indexes2, pt_muo0]))
            
            
            sample_weight_t = self.data_structure.weights_train.to_numpy().copy()
            sample_weight_v = self.data_structure.weights_val.to_numpy().copy()
            
            sample_weight = pd.DataFrame(sample_weight_t)
        
            self.err_val = sample_weight_v#np.concatenate((sample_weight_t, sample_weight_v), axis=0)
            self.sig_err = self.err_val[np.where(val_cat == "Signal")]
            self.err_val = self.err_val[np.where(val_cat != "Signal")]
            
            #self.val_cats = np.concatenate((train_cat, val_cat), axis=0)
            self.val_cats = self.val_cats[np.where(val_cat != "Signal")]
            

            #* Tuning, training, and inference
            
            self.trainModel(self.X_train, X_val, sample_weight)
            self.runInference(X_val, signal,True)
            self.checkReconError(self.channels, sig_name=f"pT_{ptscaling}")     

            
            
            
            et = time.time()
                
            try:
                img_path = Path(f"histo/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_pT_{ptscaling}.pdf")
                path = STORE_IMG_PATH/img_path

                files = {"photo":open(path, "rb")}
                message = f"Done calculating pt scaled plot with scaling {ptscaling}, took {et-st:.1f}s or {(et-st)/60:.1f}m"
                resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
                print(resp.status_code)
            except:
                pass
            
            
            
            pass 