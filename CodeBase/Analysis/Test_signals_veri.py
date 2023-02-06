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
    
    
    
class SignalDumVeri(model):
    def __init__(self,data_structure:object, path:str)->None:
        super().__init__(data_structure, path)
        
    
    def run(self):
        st = time.time()
        
        plotRMMMatrix = plotRMM(DATA_PATH, rmm_structure, 23)
        
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
            sample_weight = self.data_structure.weights_train
            
            
            self.sig_err = self.signal_weights.to_numpy()[np.where(self.signal_cats == signal_name )]
            signal = self.signal[np.where(self.signal_cats == signal_name )]
            signal_cats = self.signal_cats[np.where(self.signal_cats == signal_name )]
            event = int(np.random.choice(np.shape(signal)[0], size=1, replace=False))
            print(event, type(event))
            
            
            plotRMMMatrix.plotDfRmmMatrixNoMean(signal, signal_name[21:31], event, additional_info="signal_name[21:31]", fake=True)
            plotRMMMatrix.plotDfRmmMatrix(signal, signal_name[21:31])
            
            
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
            
            #* Reconstruction cut
            error_cut_val = np.where(self.recon_err_back > 0)[0]
            error_cut_sig = np.where(self.recon_sig > 0)[0]
            
            print(f"val cut shape: {np.shape(error_cut_val)}")
        
            trilep_mass_val = self.X_val_trilep_mass[error_cut_val]
            trilep_mass_signal = self.signal_trilep_mass[error_cut_sig]
            
            val_weights = self.err_val[error_cut_val]
            sig_weights = self.sig_err[error_cut_sig]
            
            val_cats = val_cat[error_cut_val]
            signal_cats = signal_cats[error_cut_sig]
            
            print(f"Signal error cut: {np.shape(error_cut_sig)}, trilep with cut: {np.shape(trilep_mass_signal)}")
        
            histoname = "Trilepton invariant mass for MC val and Susy signal"
            featurename = "Trilepton mass"
            PH = PlotHistogram(self.path, trilep_mass_val, val_weights, val_cats, histoname, featurename, trilep_mass_signal, sig_weights, signal_cats)
            PH.histogram(self.channels, sig_name=f"{signal_name[21:31]}", bins=25)
            
            et = time.time()
            
            
            img_path = Path(f"histo/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{signal_name[21:31]}_{featurename}.pdf")
            path = STORE_IMG_PATH/img_path

            files = {"photo":open(path, "rb")}
            message = f"Done calculating dummy signal plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
            resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
            print(resp.status_code)
            