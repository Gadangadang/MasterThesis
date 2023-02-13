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

from histo import PlotHistogram
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
    
    
    
class ETM_TRILEP(model):
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

            
            val_cat = self.data_structure.val_categories.to_numpy()
            
            self.sig_err = self.signal_weights.to_numpy()[np.where(self.signal_cats == signal_name )]
            signal = self.signal[np.where(self.signal_cats == signal_name )]
            signal_cats = self.signal_cats[np.where(self.signal_cats == signal_name )]
        
            histoname = "Trilepton invariant mass for MC val and Susy signal"
            featurename = "Trilepton mass"
            PH = PlotHistogram(self.path, self.X_val_trilep_mass, self.err_val, val_cat, histoname, featurename, self.signal_trilep_mass, self.sig_err, signal_cats)
            PH.histogram(self.channels, sig_name=f"{signal_name[21:31]}", bins=25)
            
            histoname = "Transverse missing energy for MC val and Susy signal"
            featurename = r"$E_{T}^{miss}$"
            PH = PlotHistogram(self.path, self.X_val_eTmiss, self.err_val, val_cat, histoname, featurename, self.signal_eTmiss, self.sig_err, signal_cats)
            PH.histogram(self.channels, sig_name=f"{signal_name[21:31]}", bins=25)
            
            et = time.time()
            
            img_path = Path(f"histo/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{signal_name[21:31]}_{featurename}.pdf")
            path = STORE_IMG_PATH/img_path

            files = {"photo":open(path, "rb")}
            message = f"Done calculating dummy signal plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
            resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
            print(resp.status_code)
            