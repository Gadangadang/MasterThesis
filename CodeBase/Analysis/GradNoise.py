import time
import random
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
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
        elif choice == "sample":
            signal, signame = self.sampledRMMS()
            
        self.sig_err = np.ones(np.shape(signal)[0])
        
        print(signal)
        
        self.runInference(X_val, signal, True)

        self.checkReconError(self.channels, sig_name=signame, Noise=True)
        
        et = time.time()
       
        
        img_path = Path(f"histo/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{signame}.pdf")
        path = STORE_IMG_PATH/img_path

        files = {"photo":open(path, "rb")}
        message = f"Done calculating noise plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
        resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
        print(resp.status_code)
        
    
    def sampledRMMS(self):
        flcomp_val = self.data_structure.flcomp_val.to_numpy()
        flcomp_train = self.data_structure.flcomp_train.to_numpy()
        
        
        
        if scaler == "MinMax":
            
            column_trans = ColumnTransformer(
                    [('scaler_ae', scaler, self.scalecols)],
                    remainder='passthrough'
                )
        else:
            column_trans = scaler
        
        #column_trans = scaler

        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        with strategy.scope():

            self.mu = column_trans.fit_transform(np.mean(self.data_structure.noscale_X_b_train.to_numpy(), axis=0).reshape(-1, 1))
            self.sigma = column_trans.fit_transform(np.std(self.data_structure.noscale_X_b_train.to_numpy(), axis=0).reshape(-1, 1))
        
        self.mu = np.concatenate(self.mu)
        self.sigma = np.concatenate(self.sigma)
        
        """ 
        
        
        distributions = ["Weird", "eee", "eem", "emm", "mmm", "mme", "mee"]
        choice = [-1, 0, 1, 2, 3, 4, 5]
        
        for choice, dist in zip(choice, distributions):
            print(choice, dist)
            indices = np.where(flcomp_val == choice)[0]
            self.plotRMMMatrix.plotDfRmmMatrix(self.X_val[indices,:], dist)
        
        
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
       
    
        
        lep_combo_choice = ["eee", "eem", "emm", "mmm", "mme", "mee"]
        choice = np.random.choice(lep_combo_choice, len(self.X_val), p=distributions_percent)
    
        """
        test_samples = np.random.choice(range(len(self.X_val)), 1000, replace=False)
        test_samples = self.X_val[test_samples, :]
        sample_gen_data = np.zeros((1000, np.shape(self.X_val)[1]))
        
        for index, row in enumerate(test_samples):
            samples = []
            for m, s in zip(self.mu, self.sigma):
                dist = np.abs(np.random.normal(m, 2*s))
                samples.append(dist)
                
            samples = np.asarray(samples)
            
            mask = np.where(row == 0)
            
            samples[mask] = 0
            
            sample_gen_data[index, :] = samples
            
            
            
        self.plotRMMMatrix.plotDfRmmMatrix(sample_gen_data, "masking")
    
        """dataset = []
        for combo in lep_combo_choice:
            
            combos = choice[np.where(choice == combo)]
            dataset.append(self.create_event(combos, self.mu, self.sigma))
            
        data = np.concatenate(dataset)
        """
               #print(data)
        return sample_gen_data, "Sampled_MC"
        
        
        
        

 


    def create_event(self, lep_combo:np.ndarray, average_features:np.ndarray, std_features:np.ndarray)-> np.ndarray:
        
        dataset = np.zeros((2, len(average_features))) #int(len(lep_combo)/5e5)
        print(np.shape(dataset))
        
        jets_sampling_choices = [0,1,2,3,4]
        numb_jets = np.random.choice(jets_sampling_choices, len(lep_combo))
        
        
        lep_combo_choice = np.unique(lep_combo)[0]
        
        lep_combo = {
           "eee" : ([5,6,7], 15*8, [15*8,15**2]),
           "eem" : ([5,6,10], 15*11, [15*7, 15*10], [15*11, 15**2]),
           "emm" : ([5,10,11], 15*12, [15*7, 15*10], [15*11, 15**2]),
           "mmm" : ([10,11,12], 15*13, [15*7, 15*10]),
           "mme" : ([10,11,5], 15*14, [15*5, 15*10], [15*11, 15**2]),
           "mee" : ([10,5,6], 15**2, [15*6, 15*10], [15*11, 15**2])
        }
        
        index_start = lep_combo[lep_combo_choice][0]
        limit = lep_combo[lep_combo_choice][1]
        lowerlimit = lep_combo[lep_combo_choice][2]
        try:
            upperlimit = lep_combo[lep_combo_choice][3]
        except:
            pass
        jumpp = int(np.sqrt(len(average_features)))
        
        
        
        print(f"{lep_combo_choice} data creation started")
        
        
        
        
        
        
        """
        
        for index in tqdm(range(len(dataset))):
            flag = True
            
            dataset[0,0] = np.abs(np.random.normal(average_features[0], std_features[0] ))
            number_of_jets = numb_jets[index]
            
          
            print(f"Number of jets: {number_of_jets}")
            
            for hl in range(0, limit, 15):
                if hl >= lowerlimit[0] and hl <= lowerlimit[1]:
                    continue
                try:
                    if hl >= upperlimit[0] and hl <= upperlimit[1]:
                        continue
                except:
                    pass
                dataset[index, hl] = np.abs(np.random.normal(average_features[hl], std_features[hl] ))
            
            if number_of_jets == 0:
                pass
            else:
                
                for jet in range(number_of_jets):
                    
                    
                    for jet_idx in range(jet, len(average_features), jumpp):
                        
                        if jet_idx >= limit:
                            
                            break
                        
                        if jet_idx >= lowerlimit[0] and jet_idx <= lowerlimit[1]:
                            continue
                        try:
                            if jet_idx >= upperlimit[0] and jet_idx <= upperlimit[1]:
                                continue
                        except:
                            pass
                        print(f"Jet idx: {jet_idx}")
                        
                        jet_sampling = np.abs(np.random.normal(average_features[jet_idx+1], std_features[jet_idx+1] ))
                        dataset[index, jet_idx+1] = jet_sampling #np.random.choice(jet_sampling, 1)
                       
                        

            
            for lep_index in index_start:
                
                
                for lep_idx in range(lep_index, len(average_features), jumpp):
                    
                    if lep_idx >= limit:
                        break
                    if lep_idx >= lowerlimit[0] and lep_idx <= lowerlimit[1]:
                            continue
                    try:
                        if lep_idx >= upperlimit[0] and lep_idx <= upperlimit[1]:
                            continue
                    except:
                        pass
                    print(f"Lep idx: {lep_idx}")
                  
                    lep_sampling = np.abs(np.random.normal(average_features[lep_idx], std_features[lep_idx]))
                    dataset[index, lep_idx] = lep_sampling #np.random.choice(lep_sampling, 1)
        """       
                       
        
        print(f"{lep_combo_choice} done!")
        self.plotRMMMatrix.plotDfRmmMatrix(dataset, "test_"+lep_combo_choice)
        
        return dataset
        
        
        
        
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
        
        
        
     