import time
import matplotlib
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
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
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

import seaborn as sns

plt.style.use("bmh")
sns.color_palette("hls", 1)

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



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
            print(signal_name)
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
            
            
            #plotRMMMatrix.plotDfRmmMatrixNoMean(signal, signal_name[21:31], event, additional_info="signal_name[21:31]", fake=True)
            #plotRMMMatrix.plotDfRmmMatrix(signal, signal_name[21:31])
            
            
            #* Tuning, training, and inference
            
           

            self.trainModel(self.X_train, self.X_val, sample_weight)

            
            self.runInference(self.X_val, signal, True)
            
            
            sig_name = signal_name[21:31]
        
            self.checkReconError(self.channels, sig_name=f"{sig_name}") 
            
            mean = np.mean(self.n_bins)
            std = np.std(self.n_bins)
             
            print(f"Mean recon: {mean}, std recon: {std}")
            
            #* Reconstruction cut
            error_cut_val = np.where(self.recon_err_back > (mean + std))[0]
            error_cut_sig = np.where(self.recon_sig > (mean + std))[0]
            
            print(f"val cut shape: {np.shape(error_cut_val)}")
        
            trilep_mass_val = self.X_val_trilep_mass[error_cut_val]
            trilep_mass_signal = self.signal_trilep_mass[error_cut_sig]
            
            val_weights = self.err_val[error_cut_val]
            sig_weights = self.sig_err[error_cut_sig]
            
            val_cats = val_cat[error_cut_val]
            signal_cats = signal_cats[error_cut_sig]
            
            print(f"Signal error cut: {np.shape(error_cut_sig)}, trilep with cut: {np.shape(trilep_mass_signal)}")
        
            histoname = "Trilepton invariant mass for MC val and Susy signal"
            histo_title = "Trilepton_mass"
            featurename = "Trilepton mass"
            PH = PlotHistogram(self.path, trilep_mass_val, val_weights, val_cats, histoname, featurename, histo_title, trilep_mass_signal, sig_weights, signal_cats)
            PH.histogram(self.channels, sig_name=f"{sig_name}", bins=25)
            
            
            eTmiss_val = self.X_val_eTmiss[error_cut_val]
            eTmiss_signal = self.signal_eTmiss[error_cut_sig]
            
            histoname = "Transverse missing energy for MC val and Susy signal"
            histo_title = "etmiss"
            featurename = r"$E_{T}^{miss}$"
            PH = PlotHistogram(self.path, eTmiss_val, val_weights, val_cats, histoname, featurename, histo_title, eTmiss_signal, sig_weights, signal_cats)
            PH.histogram(self.channels, sig_name=f"{sig_name}", bins=25)
            
            et = time.time()
            
            
            img_path = Path(f"histo/{LEP}/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{sig_name}_{histo_title}.pdf")
            path = STORE_IMG_PATH/img_path

            files = {"photo":open(path, "rb")}
            message = f"Done calculating dummy signal plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
            resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
            print(resp.status_code)
            
            
            #* ROC curve stuff
            
            etmiss_back = self.X_val[:, 0].copy()
            bkg = np.zeros(len(self.X_val))
            
            etmiss_sig = signal[:, 0].copy()
            sg = np.ones(len(signal))
            
            label = np.concatenate((bkg, sg))
            scores = np.concatenate((etmiss_back,etmiss_sig))
            
            scaleFactor = np.sum(self.sig_err) / np.sum(self.err_val)
            
            weights = np.concatenate((self.err_val*scaleFactor, self.sig_err))
            
            fpr, tpr, thresholds = roc_curve(label, scores, sample_weight = weights, pos_label=1)
            sorted_index = np.argsort(fpr)
            fpr =  np.array(fpr)[sorted_index]
            tpr = np.array(tpr)[sorted_index]
            
            roc_auc = auc(fpr,tpr)
            
            #RocCurveDisplay.from_predictions(label, scores, sample_weight=weights)
            plt.plot(fpr, tpr, label=f"AUC score: {roc_auc:.2f}")
            plt.xlabel("False positive rate", fontsize=25)
            plt.ylabel("True positive rate", fontsize=25)
            plt.legend(prop={"size": 15})
            plt.title(r"ROC curve of $e_T^{miss}$ for SM bkg and " + f"SUSY{sig_name}", fontsize=25)
            plt.savefig(STORE_IMG_PATH + f"histo/{LEP}/{TYPE}/{arc}/{SCALER}/roc_curve_etmiss_{sig_name}.pdf")
            plt.close()
            
            
           
            scores = np.concatenate((self.recon_err_back,self.recon_sig))
            
            fpr, tpr, thresholds = roc_curve(label, scores, sample_weight = weights, pos_label=1)
            sorted_index = np.argsort(fpr)
            fpr =  np.array(fpr)[sorted_index]
            tpr = np.array(tpr)[sorted_index]
            
            roc_auc = auc(fpr,tpr)
            
            #RocCurveDisplay.from_predictions(label, scores, sample_weight=weights)
            plt.plot(fpr, tpr, label=f"AUC score: {roc_auc:.2f}")
            plt.xlabel("False positive rate", fontsize=25)
            plt.ylabel("True positive rate", fontsize=25)
            plt.legend(prop={"size": 15})
            plt.title(f"ROC curve of recon error for SM bkg and SUSY{sig_name}", fontsize=25)
            plt.savefig(STORE_IMG_PATH + f"histo/{LEP}/{TYPE}/{arc}/{SCALER}/roc_curve_recon_err_{sig_name}.pdf")
            plt.close()
            
         