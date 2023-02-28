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

from ScaleAndPrep import ScaleAndPrep

from plotRMM import plotRMM
from Utilities.config import *
from Utilities.pathfile import *

seed = tf.random.set_seed(1)


scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}
scaler = scalers[SCALER]

if SMALL:
    arc = "small"
else:
    arc = "big"

if __name__ == "__main__":
    cols = np.load(DATA_PATH / "dfcols.npy", allow_pickle=True)
    
    idxs = []
    for col in ["flcomp", "ele_0_charge", "ele_1_charge", "ele_2_charge","muo_0_charge","muo_1_charge","muo_2_charge","nlep_BL","nlep_SG","TrileptonMass","wgt_SG"]:
        idx = np.where(cols == col)
        idxs.append(idx)
        
    cols = np.delete(cols, np.asarray(idxs))
    
    np.save(DATA_PATH / "dfcols.npy", cols)