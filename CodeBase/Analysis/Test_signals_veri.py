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
        
        file = f"3lep_significance_{TYPE}_{size_m}.txt"
        
        
        
        self.signal_cats = self.signal_cats.to_numpy()
        
        print(np.unique(self.signal_cats))
        
        val_cat = self.data_structure.val_categories.to_numpy()
        sample_weight = self.data_structure.weights_train
        
        self.trainModel(self.X_train, self.X_val, sample_weight)
        
        
        for signal in np.unique(self.signal_cats):
            
            signal_name = signal
            print(signal_name)
            
            sig_idx = np.where(self.signal_cats == signal_name )
            self.sig_err = self.signal_weights.to_numpy()[sig_idx]
            signal = self.signal[sig_idx]
            signal_cats = self.signal_cats[sig_idx]
            event = int(np.random.choice(np.shape(signal)[0], size=1, replace=False))
            print(event, type(event))
           
            #* Inference
            
            self.runInference(self.X_val, signal, True)
            sig_name = signal_name[21:31]
            self.checkReconError(self.channels, sig_name=f"{sig_name}") 
            
            histoname = "Transverse missing energy for MC val and Susy signal"
            etmiss_histoname = r"$e_T^{miss}$"
            featurename = r"$E_{T}^{miss}$"
            
            plotetmiss_cut = PlotHistogram(STORE_IMG_PATH, 
                                            self.X_val_eTmiss, 
                                            self.err_val, 
                                            val_cat, 
                                            histoname=etmiss_histoname, 
                                            featurename=r"$e_T^{miss}$",
                                            histotitle=histoname,
                                            signal=self.signal_eTmiss[sig_idx], 
                                            signal_weights=self.sig_err, 
                                            signal_cats=signal_cats)
            
            plotetmiss_cut.histogram(self.channels, sig_name=sig_name, etmiss_flag=True)
        
            #* Reconstruction cut
            string_write = f"\nSignal: {sig_name}\n"
            write_to_file(file, string_write)
            
            
            small_sig = self._significance_small(len(self.signal_eTmiss[sig_idx])*np.sum(self.sig_err), len(self.X_val_eTmiss)*np.sum(self.err_val))
            big_sig = self._significance_big(len(self.signal_eTmiss[sig_idx])*np.sum(self.sig_err), len(self.X_val_eTmiss)*np.sum(self.err_val))
            
            print(" ")
            print(f"Pre cut etmiss;  Signifance small: {small_sig} | Significance big: {big_sig}")
            print(" ")
            
            string_write = f"\nPre reconstruction error cut:\n"
            write_to_file(file, string_write)
            
            string_write = f"Significance small: {small_sig} | Signifiance big: {big_sig}\n"
            write_to_file(file, string_write)
            
            
            median = np.median(self.n_bins)
            std = np.abs(median/5)
            
            print(f"Median recon: {median}, std recon: {std}")
            
            string_write = f"\nPost recon err cut\n"
            write_to_file(file, string_write)
            
            
            #* Significance
            
            for std_scale in range(1, 4):
                
                recon_er_cut = median + std_scale*std
                print(f"Recon err cut: {recon_er_cut}")
                
                error_cut_val = np.where(self.recon_err_back > (recon_er_cut))[0]
                error_cut_sig = np.where(self.recon_sig > (recon_er_cut))[0]
                
                print(f"val cut shape: {np.shape(error_cut_val)}")
                
                trilep_mass_val = self.X_val_trilep_mass[error_cut_val]
                trilep_mass_signal = self.signal_trilep_mass[error_cut_sig]
                
                val_weights_cut = self.err_val[error_cut_val]
                sig_weights_cut = self.sig_err[error_cut_sig]
                
                val_cats_cut = val_cat[error_cut_val]
                signal_cats_cut = signal_cats[error_cut_sig]
                
                etmiss_bkg = self.X_val_eTmiss[error_cut_val]
                etmiss_sig = self.signal_eTmiss[error_cut_sig]
                
                small_sig = self._significance_small(len(etmiss_sig)*np.sum(sig_weights_cut), len(etmiss_bkg)*np.sum(val_weights_cut ))
                big_sig = self._significance_big(len(etmiss_sig)*np.sum(sig_weights_cut), len(etmiss_bkg)*np.sum(val_weights_cut ))
                
                print(" ")
                print(f"Signifance small: {small_sig} | Significance big: {big_sig}")
                print(" ")
                
                string_write = f"Recon error cut: {recon_er_cut}\n"
                write_to_file(file, string_write)
                
                string_write = f"Significance small: {small_sig} | Signifiance big: {big_sig}\n"
                write_to_file(file, string_write)
                
                
                
                histoname = "Transverse missing energy for MC val and Susy signal"
                etmiss_histoname = r"$e_T^{miss}$ with recon err cut of "+ f"{recon_er_cut:.2f}"
                featurename = r"$E_{T}^{miss}$"
                
                plotetmiss_cut = PlotHistogram(STORE_IMG_PATH, 
                                               etmiss_bkg, 
                                               val_weights_cut, 
                                               val_cats_cut, 
                                               histoname=histoname, 
                                               featurename=etmiss_histoname,
                                               histotitle=f"recon_errcut_{recon_er_cut:.2f}",
                                               signal=etmiss_sig, 
                                               signal_weights=sig_weights_cut, 
                                               signal_cats=signal_cats_cut)
                plotetmiss_cut.histogram(self.channels, sig_name=sig_name, etmiss_flag=True)
                
                histoname = "Trilepton invariant mass for MC val and Susy signal"
                histo_title = f"Trilepton mass with recon err cut of {recon_er_cut:.2f}"
                featurename = "Trilepton mass"
                PH = PlotHistogram(self.path, 
                                   trilep_mass_val, 
                                   val_weights_cut, 
                                   val_cats_cut, 
                                   histoname, 
                                   featurename, 
                                   histo_title, 
                                   trilep_mass_signal, 
                                   sig_weights_cut, 
                                   signal_cats_cut)
                PH.histogram(self.channels, sig_name=f"{sig_name}", etmiss_flag=True)
                
                
                
                
                
                
                
                    
                
            
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
            
            
            et = time.time()
    
            message = f"Done calculating dummy signal plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
            print(message)
            
            
def write_to_file(file, string_write):
    with open(file, 'a') as f:
        f.write(string_write)  