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
        
        if SMALL:
            self.checkpointname = "small"
        else:
            self.checkpointname = "big"
            
    def run(self):
        st = time.time()
        
        file = f"3lep_significance_{TYPE}_{size_m}.txt"
        
        
        
        self.signal_cats = self.signal_cats.to_numpy()
        
        print(np.unique(self.signal_cats))
        
        val_cat = self.data_structure.val_categories.to_numpy()
        sample_weight = self.data_structure.weights_train
        
        if TRAIN:
            self.trainModel(self.X_train, self.X_val, sample_weight)
            
            if TYPE == "VAE":
                    self.AE_model.encoder.save_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_{LEP}_encoder_{self.checkpointname}')
                    self.AE_model.encoder.save_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_{LEP}_decoder_{self.checkpointname}')
            else:
                self.AE_model.save_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_{LEP}_{self.checkpointname}')
        else:
            #* Load model 
            if SMALL:
                self.AE_model = self.getModel()
            else:
                self.AE_model = self.getModelBig()
                
            if TYPE == "VAE":
                    self.AE_model.encoder.load_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_{LEP}_encoder_{self.checkpointname}')
                    self.AE_model.encoder.load_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_{LEP}_decoder_{self.checkpointname}')
            else:
                self.AE_model.load_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_{LEP}_{self.checkpointname}')
                
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
            
            histoname = "etmiss_no_reconerr_cut"
            etmiss_histoname = r"$e_T^{miss}$ no reconstruction error cut"
            featurename = r"$E_{T}^{miss}$ [GeV]"
            
            plotetmiss_cut = PlotHistogram(STORE_IMG_PATH, 
                                            self.X_val_eTmiss, 
                                            self.err_val, 
                                            val_cat, 
                                            histoname=etmiss_histoname, 
                                            featurename=featurename,
                                            histotitle=histoname,
                                            signal=self.signal_eTmiss[sig_idx], 
                                            signal_weights=self.sig_err, 
                                            signal_cats=signal_cats)
            
            plotetmiss_cut.histogram(self.channels, sig_name=sig_name, etmiss_flag=True)
        
            #* Reconstruction cut
            string_write = f"\nSignal: {sig_name}\n"
            write_to_file(file, string_write)
            
            
            small_sig = self._significance_small(np.sum(self.sig_err), np.sum(self.err_val))
            big_sig = self._significance_big(np.sum(self.sig_err), np.sum(self.err_val))
            
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
                
                small_sig = self._significance_small(np.sum(sig_weights_cut), np.sum(val_weights_cut ))
                big_sig = self._significance_big(np.sum(sig_weights_cut), np.sum(val_weights_cut ))
                
                print(" ")
                print(f"Signifance small: {small_sig} | Significance big: {big_sig}")
                print(" ")
                
                string_write = f"Recon error cut: {recon_er_cut}\n"
                write_to_file(file, string_write)
                
                string_write = f"Significance small: {small_sig} | Signifiance big: {big_sig}\n"
                write_to_file(file, string_write)
                
                
                #* Etmiss tail search
                
                histoname = f"etmiss_recon_errcut_{recon_er_cut:.2f}"
                etmiss_histoname = r"$e_T^{miss}$ with recon err cut of "+ f"{recon_er_cut:.2f}"
                featurename = r"$E_{T}^{miss}$ [GeV]"
                
                
                
                plotetmiss_cut = PlotHistogram(STORE_IMG_PATH, 
                                               etmiss_bkg, 
                                               val_weights_cut, 
                                               val_cats_cut, 
                                               histoname=etmiss_histoname, 
                                               featurename=featurename,
                                               histotitle=histoname,
                                               signal=etmiss_sig, 
                                               signal_weights=sig_weights_cut, 
                                               signal_cats=signal_cats_cut)
                plotetmiss_cut.histogram(self.channels, sig_name=sig_name, etmiss_flag=True)
                
                #* Significance as function of etmiss
                if not isinstance(plotetmiss_cut.n_bins, int):
                    small_sign, big_sign  = bin_integrate_significance(plotetmiss_cut.n_bins, 
                                            etmiss_bkg, 
                                            etmiss_sig, 
                                            val_weights_cut, 
                                            sig_weights_cut)
                    
                    plt.plot(plotetmiss_cut.n_bins, small_sign,"r-", label=r"$\sqrt{2((s+b)log(1+\frac{s}{b}) - s)}$")
                    plt.plot(plotetmiss_cut.n_bins, big_sign,"b-", label=r"$\frac{s}{\sqrt{b}}$")
                    plt.legend()
                    plt.xlabel(r"$e_T^{miss}$ [GeV]", fontsize=25)
                    plt.ylabel("Signifiance", fontsize=25)
                    plt.legend(prop={"size": 15})
                    plt.title(r"Significance as function of $e_T^{miss}$", fontsize=25)
                    plt.savefig(STORE_IMG_PATH +f"histo/{LEP}/{TYPE}/{arc}/{SCALER}/significance_etmiss_{sig_name}_{recon_er_cut}.pdf")
                    plt.close()
                    
                #* Trilepton bump search
                histoname = f"Trilepton invariant mass with recon err cut of {recon_er_cut:.2f}"
                histo_title =  f"mlll_recon_errcut_{recon_er_cut:.2f}"
                featurename = r"$m_{lll}$ [GeV]"
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
        

def bin_integrate_significance(bins, dist_bkg, dist_sig, dist_bkg_weights, dist_sig_weights):
    
    sign_bin_based_small = []
    sign_bin_based_big = []
   
    for bin in bins:
        
        bin_cond = np.where(dist_bkg > bin)[0]
        bin_cond_sig = np.where(dist_sig > bin)[0]
        
        s = dist_bkg_weights[bin_cond]
        s2 = dist_sig_weights[bin_cond_sig]
        w_bins_integrated_sum_bkg =np.sum(s)
        w_bins_integrated_sum_sig= np.sum(s2)
        
        sig_small = _significance_small(w_bins_integrated_sum_sig, w_bins_integrated_sum_bkg)
        sig_big = _significance_big(w_bins_integrated_sum_sig, w_bins_integrated_sum_bkg)
        
        sign_bin_based_big.append(sig_big)
        sign_bin_based_small.append(sig_small)
        
    return sign_bin_based_small, sign_bin_based_big

def _significance_small(s, b):
        return np.sqrt(2*(( s + b )*np.log( 1 + s / b) - s ))
    
def _significance_big(s, b):
    return s / np.sqrt(b)