import sys
import time
import random
import requests
import scipy.stats
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

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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
        
        self.plotRMMMatrix = plotRMM(self.path, rmm_structure, RMMSIZE)
        self.lower = 0
        self.upper = 1
    
    
    
        
    def run(self, tune=False, train=False, choice="sigAVG")->None:
        """_summary_

        Args:
            tune (bool, optional): _description_. Defaults to False.
            train (bool, optional): _description_. Defaults to False.
            choice (str, optional): _description_. Defaults to "sigAVG".
        """
        st = time.time()
        
        
        
        rows, cols = np.shape(self.X_val)
        
        print(rows, cols)
        
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
            signal, signame = self.sampledRMMS(num_events=BACTH_SIZE*2, X_val=X_val)
            
        self.sig_err = np.ones(np.shape(signal)[0])
        
        print(np.shape(X_val), np.shape(signal))
        
        print(len(X_val)/BACTH_SIZE)
        
        
        print("Inference started")
        
        
        self.runInference(X_val, signal, True)
        
        print("Inference ended")
        

        self.checkReconError(self.channels, sig_name=signame, Noise=True)
        
        et = time.time()
       
        
        img_path = Path(f"histo/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{signame}.pdf")
        path = STORE_IMG_PATH/img_path

        files = {"photo":open(path, "rb")}
        message = f"Done calculating noise plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
        resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
        print(resp.status_code)
        
    
    def sampledRMMS(self, num_events:int, X_val:np.ndarray)->Tuple[np.ndarray, str]:
        """
        Create events based on the mean and std of the validation set. 

        Args:
            num_events (int): Number of events to create
            X_val (np.ndarray): Validation set to 

        Returns:
            Tuple[np.ndarray, str]: Sampled events for inference and the name of the set
        """
        
        
        
        
        
        """ #* Scaling the features with each other, not correct I think
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
        
        
        
        
        val_cats = self.data_structure.val_categories.to_numpy()
        ttbar_events = np.where(val_cats=="ttbar")[0]
        
        test_samples_events = np.random.choice(ttbar_events, num_events, replace=False)
        test_samples = X_val[test_samples_events, :]
        self.mu = np.mean(test_samples, axis=0)
        self.sigma = np.std(test_samples, axis=0)
        
        sample_gen_data = np.zeros((np.shape(test_samples)))
        
        print("Sample gen data shape", np.shape(sample_gen_data))
        
        
        p = 0
        samples = np.zeros((len(test_samples), len(self.mu)))
        for m, s in zip(self.mu, self.sigma):
            
            
            
            choice_m = np.random.choice([-1, 1], 1)
            choice_s = np.random.choice([-1, 1], 1)
            
            m += choice_m*0.2
            m = np.abs(m)
            if s == 0:
                s = 2*sys.float_info.epsilon
                s += choice_s*2e-17
            
            dist = scipy.stats.truncnorm.rvs((self.lower-m)/s,(self.upper-m)/s,loc=m,scale=s,size=len(test_samples)) #np.abs(np.random.normal(m, s))
            samples[:, p] = dist
            p+=1
        
        #samples = scipy.stats.truncnorm.rvs((self.lower-self.mu)/self.sigma,(self.upper-self.mu)/self.sigma,loc=self.mu,scale=self.sigma,size=np.shape(sample_gen_data)) #np.abs(np.random.normal(m, s))
            
        for index in tqdm(range(len(test_samples))):
            
            mask = np.where(test_samples[index, :] == 0)[0]
            samples[index, mask] = 0
            sample_gen_data[index, :] = samples[index, :]
            #self.plotRMMMatrix.plotDfRmmMatrixNoMean(sample_gen_data, f"ttbar", index)
            
            
        
            
        self.plotRMMMatrix.plotDfRmmMatrix(sample_gen_data, "masking")
    
               #print(data)
        return sample_gen_data, "Sampled_MC"
        
        
        
        

 


    def create_event(self, lep_combo:np.ndarray, average_features:np.ndarray, std_features:np.ndarray)-> np.ndarray:
        """_summary_

        Args:
            lep_combo (np.ndarray): _description_
            average_features (np.ndarray): _description_
            std_features (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        
        
        flcomp_val = self.data_structure.flcomp_val.to_numpy()
        flcomp_train = self.data_structure.flcomp_train.to_numpy()
        
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
        
        
        
        dataset = []
        for combo in lep_combo_choice:
            
            combos = choice[np.where(choice == combo)]
            dataset.append(self.create_event(combos, self.mu, self.sigma))
            
        data = np.concatenate(dataset)
    
        
        
        
        
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
   
                       
        
        print(f"{lep_combo_choice} done!")
        self.plotRMMMatrix.plotDfRmmMatrix(dataset, "test_"+lep_combo_choice)
        
        return dataset
        
        
        
        
    def sigAvgBasedOnMC(self, X_train:np.ndarray, X_val:np.ndarray)->Tuple[np.ndarray, str]:
        """_summary_

        Args:
            X_train (np.ndarray): _description_
            X_val (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, str]: _description_
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
        
        
        
     