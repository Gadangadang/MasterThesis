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



class GradNoise(RunAE):
    def __init__(self, data_structure:object, path:str)->None:
        super().__init__(data_structure, path)  
        
        self.plotRMMMatrix = plotRMM(self.path, rmm_structure, 15)
        
    def createMuonLikeEvents(self, amount:int)-> np.ndarray:
        pass
    
    def createElectronLikeEvents(self, amount:int)-> np.ndarray:
        pass
    
    
        
    def run(self, tune=False, train=False, choice="sigAVG"):
        """_summary_

        Args:
            tune (bool, optional): _description_. Defaults to False.
            train (bool, optional): _description_. Defaults to False.
            choice (str, optional): _description_. Defaults to "sigAVG".
        """
        st = time.time()
        
        
        
        rows, cols = np.shape(self.X_val)
        
        X_train = self.X_train 
        X_val = self.X_val 
        sample_weight = self.data_structure.weights_train.to_numpy()
        sample_weight = pd.DataFrame(sample_weight)
        self.err_val = self.data_structure.weights_val.to_numpy()
        
        self.sig_err = np.ones(rows)
        
        
        
        #* Tuning, training, and inference
        if tune:
            HPT = HyperParameterTuning(self.data_structure, STORE_IMG_PATH)
            HPT.runHpSearch(
                X_train, X_val, sample_weight, small=SMALL
            )
            
            self.AE_model = HPT.AE_model
            
            self.trainModel(X_train, X_val, sample_weight)
            
        else:
            self.AE_model = tf.keras.models.load_model(
                    "tf_models/" + "model_test.h5"
                )
        
        if train:  
            self.trainModel(X_train, X_val, sample_weight)
        
       
           
        
        
        
        if choice == "sigAVG":
            signal, signame = self.sigAvgBasedOnMC(X_train, X_val) 
        
        
        self.runInference(X_val, signal, True)

        self.checkReconError(self.channels, sig_name=signame, Noise=True)
        
        et = time.time()
       
        
        img_path = Path(f"histo/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{signame}.pdf")
        path = STORE_IMG_PATH/img_path

        files = {"photo":open(path, "rb")}
        message = f"Done calculating noise plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
        resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
        print(resp.status_code)
        
    
    def genRmmEvent(self):
        flcomp_val = self.data_structure.flcomp_val.to_numpy()
        flcomp_train = self.data_structure.flcomp_train.to_numpy()
        
        self.mu = np.mean(self.X_train, axis=0)
        self.sigma = np.std(self.X_train, axis=0)
        
        
        distributions = ["Weird", "eee", "eem", "emm", "mmm", "mme", "mee"]
        choice = [-1, 0, 1, 2, 3, 4, 5]
        """
        for choice, dist in zip(choice, distributions):
            print(choice, dist)
            indices = np.where(flcomp_val == choice)[0]
            self.plotRMMMatrix.plotDfRmmMatrix(self.X_val[indices,:], dist)
        """ 
        
        non_weird = np.where(flcomp_val != -1)[0]
        new = flcomp_val[non_weird]

        distributions_percent = [
            len(np.where(new == 0)[0])/len(new),
            len(np.where(new == 1)[0])/len(new),
            len(np.where(new == 2)[0])/len(new),
            len(np.where(new == 3)[0])/len(new),
            len(np.where(new == 4)[0])/len(new),
            len(np.where(new == 5)[0])/len(new)
        ]
        
        print(np.sum(distributions_percent))
       
        
        
        print(self.data_structure.cols[:15])
        
        jets_sampling_choices = [0,1,2,3,4]
        
        lep_combo = {
           "eee" : [5,6,7],
           "eem" : [5,6,10],
           "emm" : [5,10,11],
           "mmm" : [10,11,12],
           "mme" : [10,11,5],
           "mee" : [10,5,6]
        }
        
        lep_combo_choice =["eee", "eem", "emm", "mmm", "mme", "mee"]
        choice = np.random.choice(lep_combo_choice, len(self.X_val), p=distributions_percent)
        
        sample_gen_data = np.zeros_like(self.X_val)
        print(np.shape(sample_gen_data), np.shape(self.X_val))


        
        
        
        
    def sigAvgBasedOnMC(self, X_train, X_val):
        """_summary_

        Args:
            X_train (_type_): _description_
            X_val (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.mu = np.mean(X_train, axis=0)
        self.sigma = np.std(X_train, axis=0)
        
        signal = []
        
        Feat = 0
        for si, m in zip(self.sigma, self.mu):
            if m == 0:
                print(f"self.mu 0 for feat nr {Feat}")
                sig_val = np.zeros(np.shape(X_val)[0])
            else:
                sig_val = np.abs(np.random.normal(m, si, np.shape(X_val)[0]))
            signal.append(sig_val)
            
            Feat += 1
            
        signal = np.asarray(signal).reshape(np.shape(X_val))
        
        
        
        #signal = np.random.uniform(0,1, np.shape(X_val))
        
        #print(len(np.where(signal < 0.0001)[0]), len(np.where(signal < 0.001)[0]))
        
        signal[:, np.where(self.mu == 0.)] = 0.
        
        signame = "AVG_MCBased_Noise"
        
        for i in [5, 444, 3342, 10031]:
            self.plotRMMMatrix.plotDfRmmMatrixNoMean(signal, signame, i)
            
        return signal, signame
        
        
        
     