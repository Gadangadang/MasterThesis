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


from Utilities.config import *
from Utilities.pathfile import *


seed = tf.random.set_seed(1)

tf.keras.utils.get_custom_objects()["leaky_relu"] = tf.keras.layers.LeakyReLU()


scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}
scaler = scalers[SCALER]

if SMALL:
    arc = "small"
else:
    arc = "big"





class PlotHistogram:
    def __init__(self, 
                 path,
                 err_val, 
                 err_val_weights, 
                 val_cats, 
                 histoname="Reconstruction error histogram with MC", 
                 featurename="Log10 Reconstruction Error",
                 histotitle="",
                 signal=[], 
                 signal_weights=[], 
                 signal_cats=[], 
                 data=[], 
                 data_weights=[], 
                 ):
        
        self.path = path
        self.err_val = err_val
        self.err_val_weights = err_val_weights
        self.val_cats = val_cats
        
        print(len(signal))
        print(len(signal_weights))
        print(len(signal_cats))
        
        if len(signal) != 0:
            self.signal = signal
            self.signal_weights = signal_weights
            self.signal_cats = signal_cats
            
            self.flag = 0
        self.flag = 1
            
        if len(data) != 0:
            self.data = data
            self.data_weights = data_weights
            
        self.histoname = histoname
        self.featurename = featurename
        self.histotitle = histotitle
        
        
        
     
        
        
    def histogram(self, channels: list, sig_name="nosig", bins=40, etmiss_flag=False)->None:
        """_summary_

        Args:
            channels (list): List containing all the channels
            sig_name (str, optional): Name of signal sample. Defaults to "nosig".
            Noise (bool, optional): Noise paramter, sets the bins to 25 as default. Defaults to False.
        """

        histo_atlas = []
        weight_atlas_data = []
        
        print("histo started")
        
        colors = []
        if LEP == "Lep2":
            color_schemes = colors_scheme_2lep
        elif LEP == "data":
            color_schemes = color_schemes_data
        else:
            color_schemes = colors_scheme
        
        for channel in channels:
            condition = np.where(self.val_cats == channel)[0]
            err = self.err_val[condition]
            histo_atlas.append(err)

            err_w = self.err_val_weights[condition]
           
            weight_atlas_data.append(err_w)
            colors.append(color_schemes[channel])
       
        try:
            try:
                sig_err = self.signal
                sig_err_w = self.signal_weights
            except:
                sig_err = self.data
                sig_err_w = self.data_weights
        except:
            print("No signal")
            
            
        
   

        sum_w = [np.sum(weight) for weight in weight_atlas_data]
        sort_w = np.argsort(sum_w, kind="mergesort")

        sns.set_style("darkgrid")
        plt.rcParams["figure.figsize"] = (12, 9)

        fig, ax = plt.subplots()

        try:
            N, bins = np.histogram(sig_err, bins=bins, weights=sig_err_w)
            x = (np.array(bins[0:-1]) + np.array(bins[1:])) / 2
            ax.scatter(x, N, marker="+", label=f"{sig_name}", color="black")  # type: ignore
            n_bins = bins
            print("Bins: ",n_bins)
            self.n_bins = n_bins
        except:
            self.n_bins = 25
        
        if len(colors) != len(histo_atlas):
            colors = colors[:len(histo_atlas)]
        
        
            
        
        
        if len(histo_atlas) != 1:
            
            data_histo = np.asarray(histo_atlas, dtype=object)[sort_w]
            we = np.asarray(weight_atlas_data, dtype=object)[sort_w]
            colors = np.asarray(colors, dtype=object)[sort_w]
            labels = np.asarray(channels, dtype=object)[sort_w]
        else:
            data_histo = histo_atlas
            we = weight_atlas_data
            labels = channels
        
        
        ax.hist(
            data_histo,
            self.n_bins,
            density=False,
            stacked=True,
            alpha=0.5,
            histtype="bar",
            color=colors,
            label=labels,
            weights=we,
        )

        ax.legend(prop={"size": 15})
        if data:
            ax.set_title(
                self.histoname + "and ATLAS data", fontsize=25
            )
        else:
            ax.set_title(
                self.histoname , fontsize=25
            )
        ax.set_xlabel(self.featurename, fontsize=25)
        ax.set_ylabel("#Events", fontsize=25)
        if etmiss_flag:
            ax.set_xlim([0, 1300])
        ax.set_ylim(bottom=0.1)  # type: ignore
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=25)
        fig.tight_layout()
        
        plt.savefig(self.path + f"histo/{LEP}/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{sig_name}_{self.histotitle}.pdf")
        plt.close()
        
    def histogram_data(self, channels, sig_name="nosig", bins=40, etmiss_flag=False)->None:
        
        histo_atlas = self.err_val
        weight_atlas_data = self.err_val_weights
        
        
        
        color_schemes = color_schemes_data
       
        
        colors = color_schemes[channels[0]]
       
        try:
            try:
                sig_err = self.signal
                sig_err_w = self.signal_weights
            except:
                sig_err = self.data
                sig_err_w = self.data_weights
        except:
            print("No signal")
            
            
        
   

        """sum_w = [np.sum(weight) for weight in weight_atlas_data]
        sort_w = np.argsort(sum_w, kind="mergesort")"""

        sns.set_style("darkgrid")
        plt.rcParams["figure.figsize"] = (12, 9)

        fig, (ax, ax2) = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})

        try:
            N, bins = np.histogram(sig_err, bins=bins, weights=sig_err_w)
            x = (np.array(bins[0:-1]) + np.array(bins[1:])) / 2
            ax.scatter(x, N, marker="+", label=f"{sig_name}", color="black")  # type: ignore
            n_bins = bins
            print("Bins: ",n_bins)
            self.n_bins = n_bins
        except:
            self.n_bins = 25
        
       
            
        data_histo = np.asarray(histo_atlas, dtype=object)#[sort_w]
        we = np.asarray(weight_atlas_data, dtype=object)#[sort_w]
        #colors = np.asarray(colors, dtype=object)#[sort_w]
        #labels = np.asarray(channels, dtype=object)#[sort_w]
        
        print(colors)
        
        
        ns, bins, patche = ax.hist(
            data_histo,
            self.n_bins,
            alpha=0.5,
            histtype="bar",
            color=colors,
            label=channels[0],
            weights=we,
        )

        ax.legend(prop={"size": 15})
        if data:
            ax.set_title(
                self.histoname + "and ATLAS data", fontsize=25
            )
        else:
            ax.set_title(
                self.histoname , fontsize=25
            )
        ax.set_xlabel(self.featurename, fontsize=25)
        ax.set_ylabel("#Events", fontsize=25)
        if etmiss_flag:
            ax.set_xlim([0, 1300])
        ax.set_ylim(bottom=0.1)  # type: ignore
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=25)
        fig.tight_layout()
        
        ratio = np.divide(N,
                  ns,
                  where=(ns != 0))
        
        
        
        try:
            ax2.scatter(x, ratio, marker="+", color="black")
            
            ax2.set_ylabel('Ratio (Blind/Data)', fontsize=15)
            #ax2.set_xlabel('Log10 Reconstruction error', fontsize=25)
            ax2.set_ylim([0.8, np.max(ratio)])
            ax2.tick_params(axis="both", labelsize=25)
        except:
            pass
        plt.savefig(self.path + f"histo/data/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{sig_name}_{self.histotitle}.pdf")
        plt.close()
        
        
        
        