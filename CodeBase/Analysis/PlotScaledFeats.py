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

if TYPE == "VAE":
    model = RunVAE
elif TYPE == "AE":
    model = RunAE


seed = tf.random.set_seed(1)

tf.keras.utils.get_custom_objects()["leaky_relu"] = tf.keras.layers.LeakyReLU()


scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}
scaler = scalers[SCALER]

if SMALL:
    arc = "small"
else:
    arc = "big"
    
    
    
class VizualizeFeats(model):
    def __init__(self, data_structure:object, path:str)->None:
        super().__init__(data_structure, path)  
        
    def plotfeats(self):
        feat_names = self.data_structure.cols
        
        for index, name in enumerate(feat_names):
            data = self.X_train[:, index]
            
            plt.plot(data, label=f"{name}")
            plt.savefig(self.path + f"feat_scaled/{SCALER}/feature_{name}.pdf")
            plt.close()
        
    
if __name__ == "__main__":
    pass