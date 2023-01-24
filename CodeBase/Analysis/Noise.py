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
class NoiseTrial(model):
    def __init__(self, data_structure:object, path:str)->None:
        super().__init__(data_structure, path)    
        
    def run(self):
        """
        [Summary]
        
        """
        plotRMMMatrix = plotRMM(self.path, rmm_structure, 15)
        
        st = time.time()
        
        rows, cols = np.shape(self.X_val)
        
        X_train = self.X_train 
        X_val = self.X_val 
        sample_weight = self.data_structure.weights_train.to_numpy()
        sample_weight = pd.DataFrame(sample_weight)
        self.err_val = self.data_structure.weights_val.to_numpy()
        
        self.sig_err = np.ones(rows)
        sigma = 0.1
        mu = 0.5
        s = np.abs(np.random.normal(mu, sigma, (rows, cols)))
           
        print(" ")
        print(f"{(s.nbytes)/1000000000} GBytes")
        print(" ")
        signal = s
        
        #plotRMMMatrix.plotDfRmmMatrixNoMean(s, "Noise", 0)
        
        
        #* Tuning, training, and inference
        if TYPE == "AE":
            HPT = HyperParameterTuning(self.data_structure, STORE_IMG_PATH)
            HPT.runHpSearch(
                X_train, X_val, sample_weight, small=SMALL
            )
        
            """self.AE_model = tf.keras.models.load_model(
                        "tf_models/" + "model_test.h5"
                    )"""
                    
            
            self.AE_model = HPT.AE_model

        self.trainModel(X_train, X_val, sample_weight)

        
        self.runInference(X_val, signal, True)

        self.checkReconError(self.channels, sig_name="Noise", Noise=True)
        
        et = time.time()
       
        
        img_path = Path(f"histo/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_Noise.pdf")
        path = STORE_IMG_PATH/img_path

        files = {"photo":open(path, "rb")}
        message = f"Done calculating noise plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
        resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
        print(resp.status_code)
        
        
        