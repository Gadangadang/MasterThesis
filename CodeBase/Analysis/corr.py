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

class CorrCheck(RunAE):
    def __init__(self, data_structure:object, path:str)->None:
        super().__init__(data_structure, path)   


    def checkCorr(self):
        print("Corrolation initiated")
        df = pd.DataFrame(self.X_val, columns=self.data_structure.cols)
        
        
        matrix = df.corr().round(2)
        plt.figure(figsize=(30,30)) 
        sns.heatmap(matrix, cmap='RdYlGn_r')
        plt.savefig(self.path + f"corrolation/{SCALER}/corr_train.pdf")
        plt.close()