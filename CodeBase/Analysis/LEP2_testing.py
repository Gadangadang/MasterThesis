import re
import os
import time
import random
import requests
import numpy as np
#import polars as pl 
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
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from plotRMM import plotRMM
from histo import PlotHistogram

from Utilities.config import *
from Utilities.pathfile import *

seed = tf.random.set_seed(1)


scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}
scaler = scalers[SCALER]

if SMALL:
    arc = "small"
else:
    arc = "big"
    
    
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
class RunVAE:
    
    def __init__(self, data_shape):
        self.data_shape = data_shape
    
    def getModelBig(self):
        """_summary_

        Returns:
            tf.python.keras.engine.functional.Functional: Model to use
        """
        
        val = 150
        
        encoder_inputs = tf.keras.Input(shape=self.data_shape)
        x = tf.keras.layers.Dense(units=self.data_shape, activation="relu")(encoder_inputs)
        x = tf.keras.layers.Dense(units=400, activation="tanh",)(encoder_inputs)
        x = tf.keras.layers.Dense(units=300, activation="relu")(x)
        x = tf.keras.layers.Dense(200, activation=tf.keras.layers.LeakyReLU(alpha=0.3),)(x)
        
        z_mean = tf.keras.layers.Dense(val, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(val, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        
        latent_inputs = tf.keras.Input(shape=val)
        x = tf.keras.layers.Dense(units=200, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(latent_inputs)
        x = tf.keras.layers.Dense(units=300, activation="relu",)(x)
        x = tf.keras.layers.Dense(units=400, activation="tanh",)(x)
        decoder_outputs = tf.keras.layers.Dense(units=self.data_shape, activation="sigmoid")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
            

        self.AE_model = VAE(encoder, decoder)
        self.AE_model.compile(optimizer=tf.keras.optimizers.Adam())
        #self.AE_model.save("tf_models/untrained_big_vae")
        return self.AE_model
    
    def getModel(self):
        """_summary_

        Returns:
            tf.python.keras.engine.functional.Functional: Model to use
        """
        
        val = 150
        
        encoder_inputs = tf.keras.Input(shape=self.data_shape)
        x = tf.keras.layers.Dense(units=self.data_shape, activation="relu")(encoder_inputs)
        
        z_mean = tf.keras.layers.Dense(val, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(val, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        
        latent_inputs = tf.keras.Input(shape=val)
        
        decoder_outputs = tf.keras.layers.Dense(units=self.data_shape, activation="sigmoid")(latent_inputs)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
            

        self.AE_model = VAE(encoder, decoder)
        self.AE_model.compile(optimizer=tf.keras.optimizers.Adam())
        #self.AE_model.save("tf_models/untrained_small_vae")
        
        return self.AE_model
    
class RunAE:
    def __init__(self, data_shape):
        self.data_shape = data_shape

    def getModel(self):
        """_summary_

        Returns:
            tf.python.keras.engine.functional.Functional: Model to use
        """
        # Input layer
        inputs = tf.keras.layers.Input(shape=self.data_shape, name="encoder_input")

        # First hidden layer
        x = tf.keras.layers.Dense(
            units=self.data_shape,
            activation="tanh",
            
        )(inputs)
        
        # Latent space
        val = 150
        x2 = tf.keras.layers.Dense(units=val, 
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)
        )(x)

        # Encoder definition
        encoder = tf.keras.Model(inputs, x2, name="encoder")
        encoder.summary()

        # Latent input for decoder
        latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")

        # Output layer
        output = tf.keras.layers.Dense(self.data_shape, 
                                       activation=tf.keras.layers.LeakyReLU(alpha=0.3)
        )(latent_input)

        # Decoder definition
        decoder = tf.keras.Model(latent_input, output, name="decoder")
        decoder.summary()

        # Output definition
        outputs = decoder(encoder(inputs))

        # Model definition
        AE_model = tf.keras.Model(inputs, outputs, name="AE_model")

        hp_learning_rate = 0.0015
        optimizer = tf.keras.optimizers.Adam(hp_learning_rate)
        AE_model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
        #AE_model.save("tf_models/untrained_small_ae")
        
        # tf.keras.utils.plot_model(AE_model, to_file=path+"ae_model_plot.pdf", show_shapes=True, show_layer_names=True, expand_nested=True)

        return AE_model

    def getModelBig(self):
        """_summary_

        Returns:
            tf.python.keras.engine.functional.Functional: Model to use
        """
        # Input layer
        inputs = tf.keras.layers.Input(shape=self.data_shape, name="encoder_input")

        # First hidden layer
        x = tf.keras.layers.Dense(
            units=529,
            activation="tanh",
            
        )(inputs)
        x__ = tf.keras.layers.Dense(
            units=450,
            activation="tanh",
            
        )(x)

        # Second hidden layer
        x_ = tf.keras.layers.Dense(units=300, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x__)

        # Third hidden layer
        x1 = tf.keras.layers.Dense(
            units=200,
            activation="relu",
            
        )(x_)

        val = 150

        # Forth hidden layer
        x2 = tf.keras.layers.Dense(
            units=val, activation=tf.keras.layers.LeakyReLU(alpha=0.3)
        )(x1)

        # Encoder definition
        encoder = tf.keras.Model(inputs, x2, name="encoder")
        encoder.summary()

        # Latent space
        latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")

        # Fifth hidden layer
        x = tf.keras.layers.Dense(
            units=200,
            activation="relu",
            
        )(latent_input)

        # Sixth hidden layer
        x_ = tf.keras.layers.Dense(
            units=350, activation=tf.keras.layers.LeakyReLU(alpha=0.3)
        )(x)

        # Seventh hidden layer
        x1 = tf.keras.layers.Dense(
            units=450,
            activation="tanh",
            
        )(x_)

        # Output layer
        output = tf.keras.layers.Dense(self.data_shape, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x1)

        # Decoder definition
        decoder = tf.keras.Model(latent_input, output, name="decoder")
        decoder.summary()

        # Output definition
        outputs = decoder(encoder(inputs))

        # Model definition
        AE_model = tf.keras.Model(inputs, outputs, name="AE_model")

        hp_learning_rate = 0.0015
        optimizer = tf.keras.optimizers.Adam(hp_learning_rate)
        AE_model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
        #AE_model.save("tf_models/untrained_big_ae")
        # tf.keras.utils.plot_model(AE_model, to_file=path+"ae_model_plot.pdf", show_shapes=True, show_layer_names=True, expand_nested=True)

        return AE_model
    
class LEP2ScaleAndPrep:
    def __init__(self, path: Path, event_rmm=False, save=False, load=False, lep=3, convert=True) -> None:
        """_summary_

        Args:
            path (str): _description_
            event_rmm (bool, optional):
        """
        self.path = path
        self.lep = lep
        self.onlyfiles = self.getDfNames()
        self.event_rmm = event_rmm
        self.load = load
        self.save = save
        self.convert = convert
        # self.scaleAndSplit()
        self.totmegasets = 10
        
        self.epochs = EPOCHS
        self.b_size = BACTH_SIZE
        
        self.channels = [
            "Zttjets", 
            "Wjets", 
            "singletop", 
            "ttbar", 
            "Zeejets", 
            "Zmmjets",
            "Diboson"
        ]
        
        self.signal = np.load(DATA_PATH / "signal.npy")
        self.signal_weights = pd.read_hdf(DATA_PATH / "signal_weight_b.h5")
        self.signal_categories = pd.read_hdf(DATA_PATH / "signal_cat_b.h5")
        
        if SMALL:
            self.checkpointname = "small"
        else:
            self.checkpointname = "big"
        

        

    def getDfNames(self) -> Tuple[str, ...]:
        """
        Fetches all objects in a directory

        Returns:
            Tuple[str, ...]: list of pathnames
        """
        
        if self.lep == 3:
            files = [
                f
                for f in listdir(self.path)
                if isfile(join(self.path, f))
                and f[-4:] != ".npy"
                and f[-4:] != ".csv"
                and f[-5:] != "_b.h5"
                and f[-4:] != ".txt"
                and f[-3:] != ".h5"
                and f[0:3] != "two"
                and f[-8:] != ".parquet"
            ]
        elif self.lep == 2:
            files = [
                f
                for f in listdir(self.path)
                if isfile(join(self.path, f))
                and f[-4:] != ".npy"
                and f[-4:] != ".csv"
                and f[-5:] != "_b.h5"
                and f[-4:] != ".txt"
                and f[-3:] != ".h5"
                and f[0:3] == "two"
                and f[-8:] != ".parquet"
            ]
        
      
        return files  # type: ignore
    
    def convertParquet(self):
        """
        Converts all hdf5 files to parquet files for use of polars later
        """
        
        if self.convert:
            
            
            for file in self.onlyfiles:
                
                print(f"Converting {file}")
                
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
                with strategy.scope():
                    start = file.find("two_")
                    stop = file.find("_3lep")
                    name = file[start+4:stop]
                    if name in ["data15", "data16", "data17","data18","singletop","Diboson","Zeejets1","Zeejets2","Zeejets3","Zmmjets1","Zmmjets2","Zmmjets3""Zttjets","Wjets","ttbar","Zeejets4"]:
                        continue
                    
                    df = pd.read_hdf(self.path/file)
                    
                    
                    #scaled_df = scaler.fit_transform(df)
                    name = "twolep_" + name +".parquet"
                    
                    df.to_parquet(self.path/name)
                    print(f"{name} done")
                    print(" ")
                
            

            
        
                
        self.parqs = [f for f in listdir(self.path) if isfile(join(self.path, f)) and f[-8:] == ".parquet"]
        
        print(self.parqs)

    def createMCSubsamples(self):
        """
        Create subsets that maintains the SM MC distribution. 
        
        """
        
        try:
            self.parqs
        except:
            self.convertParquet()
        
        
       
        
        print("Subsampling started")
        for file in self.parqs:
            if file[7:11] == "data":
                continue
            
            
            
            start = file.find("twolep_")
            end = file.find(".parquet")
            
            name = file[start+7:end]
            
            print(f"Subsampling channel: {name}")
            

            df = pl.read_parquet(self.path / file, use_pyarrow=True)
            
            df = df.drop(
                    [
                        "nlep_BL",
                        "nlep_SG",
                        "ele_0_charge",
                        "ele_1_charge",
                        "ele_2_charge",
                        "muo_0_charge",
                        "muo_1_charge",
                        "muo_2_charge",
                        "TrileptonMass",
                    ],
                    
                )
            
            x_b_train, x_b_val = train_test_split(
                        df, test_size=0.2, random_state=seed
            )
            
            self.sampleSet(x_b_train, x_b_val, name)
            
        print("Subsampling done")
            
            

        
        
    def sampleSet(self, xtrain, xval, name):
        
        
        
        count1 = len(xtrain)
        count2 = len(xval)
        names_train = [name] * count1
        names_train = np.asarray(names_train)
        
        
        names_val = [name] * count2
        names_val = np.asarray(names_val)
        
        #* Sample from training and validation set
        indices_train = np.asarray(range(len(xtrain)))
        np.random.shuffle(indices_train)
        split_idx_train = np.array_split(indices_train, self.totmegasets)
    
        indices_val = np.asarray(range(len(xval)))    
        np.random.shuffle(indices_val)
        split_idx_val = np.array_split(indices_val, self.totmegasets)
        
        
        weights_train_tot = xtrain["wgt_SG"].to_numpy()
        weights_val_tot = xval["wgt_SG"].to_numpy()
        
        xtrain = xtrain.drop(["flcomp", "wgt_SG"],)
        xval = xval.drop(["flcomp", "wgt_SG"],)
        
        
        megaset = 0
        for idx_set_train, idx_set_val in zip(split_idx_train, split_idx_val):
            print(f"name: {name}; megaset: {megaset}")
            
            weights_train = weights_train_tot[idx_set_train]
            weights_val = weights_val_tot[idx_set_val]

            train_categories = names_train[idx_set_train]
            val_categories = names_val[idx_set_val]
            
            #* Save weights and categories 
            np.save(DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_weights_train", weights_train)
            np.save(DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_weights_val", weights_val)
            
            np.save(DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_categories_train", train_categories)
            np.save(DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_categories_val", val_categories)
            
            #* Save the actual dataframe
            np.save(DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_x_train", xtrain.to_numpy()[idx_set_train])
            np.save(DATA_PATH / f"Megabatches/MB{megaset}/MSET{megaset}_{name}_x_val", xval.to_numpy()[idx_set_val])
            
            megaset += 1
            
            
    def mergeMegaBatches(self):
        
        print(" ")
        print(f"Megabatch merging started")
        
        for megaset in range(self.totmegasets):
            print(f"Running merging on megabatch: {megaset}")
            FETCH_PATH = DATA_PATH/ f"Megabatches/MB{megaset}"
            MERGE_PATH = FETCH_PATH / f"MergedMB{megaset}"
            
            xtrains = [np.load(FETCH_PATH/file)[:, :-1] for file in os.listdir(FETCH_PATH) if "x_train" in file]
            xtrain_weights = [np.load(FETCH_PATH/filename) for filename in os.listdir(FETCH_PATH) if "weights_train" in filename]
            xtrain_categories = [np.load(FETCH_PATH/filename) for filename in os.listdir(FETCH_PATH) if "categories_train" in filename]
           
            xvals = [np.load(FETCH_PATH/filename)[:, :-1] for filename in os.listdir(FETCH_PATH) if "x_val" in filename]
            xval_weights = [np.load(FETCH_PATH/filename) for filename in os.listdir(FETCH_PATH) if "weights_val" in filename]
            xval_categories = [np.load(FETCH_PATH/filename) for filename in os.listdir(FETCH_PATH) if "categories_val" in filename]
   
            xtrain = np.concatenate((xtrains), axis=0)
            x_train_cats = np.concatenate((xtrain_categories), axis=0)
            x_train_weights = np.concatenate((xtrain_weights), axis=0)
            
            print("Train shapes")
            [print(np.shape(array)) for array in xtrains]
            print(np.sum([np.shape(array)[0] for array in xtrains]))
            print(np.shape(xtrain))
            
            xval = np.concatenate((xvals), axis=0)
            x_val_cats = np.concatenate((xval_categories), axis=0)
            x_val_weights = np.concatenate((xval_weights), axis=0)
            
            print("Val shapes")
            [print(np.shape(array)) for array in xvals]
            print(np.sum([np.shape(array)[0] for array in xvals]))
            print(np.shape(xval))
            
            print(" ")
            print(f"xtrain: {(xtrain.nbytes)/1000000000} Gbytes")
            print(f"xval: {(xval.nbytes)/1000000000} Gbytes")
            print(" ")
            
            #* Scaling 
            self.column_trans = scaler
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            with strategy.scope():

                xtrain = self.column_trans.fit_transform(xtrain)
                xval = self.column_trans.transform(xval)
                
            
            np.save(MERGE_PATH / f"Merged{megaset}_xtrain", xtrain)
            np.save(MERGE_PATH / f"Merged{megaset}_weights_train", x_train_weights)
            np.save(MERGE_PATH / f"Merged{megaset}_categories_train", x_train_cats)
            np.save(MERGE_PATH / f"Merged{megaset}_xval", xval)
            np.save(MERGE_PATH / f"Merged{megaset}_weights_val", x_val_weights)
            np.save(MERGE_PATH / f"Merged{megaset}_categories_val", x_val_cats)
            
        print("Megabatching done")
        print(" ")
        
    
            
    def RunTraining(self):
        global data_shape
        
        if TYPE == "VAE":
            nn_model = RunVAE(data_shape=data_shape)
        elif TYPE == "AE":
            nn_model = RunAE(data_shape=data_shape)
            
        start = time.time()
        
        for megaset in range(self.totmegasets):
            start1 = time.time()
            print(f"Running training on megabatch: {megaset}")
            FETCH_PATH = DATA_PATH/ f"Megabatches/MB{megaset}"
            MERGE_PATH = FETCH_PATH / f"MergedMB{megaset}"
            
            xtrain = np.load(MERGE_PATH / f"Merged{megaset}_xtrain.npy")
            xval = np.load(MERGE_PATH / f"Merged{megaset}_xval.npy")
            x_train_weights = pd.DataFrame(np.load(MERGE_PATH / f"Merged{megaset}_weights_train.npy"))
            
            #* Load model 
            if SMALL:
                self.AE_model = nn_model.getModel()
            else:
                self.AE_model = nn_model.getModelBig()
            
            if megaset != 0:
                if TYPE == "VAE":
                    self.AE_model.encoder.load_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_encoder_{self.checkpointname}')
                    self.AE_model.encoder.load_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_decoder_{self.checkpointname}')
                else:
                    
                    self.AE_model.load_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}{self.checkpointname}')
                
            
            #* Run Training
            self._trainloop(xtrain, xval, x_train_weights)
                
            if TYPE == "VAE":
                self.AE_model.encoder.save_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_encoder_{self.checkpointname}')
                self.AE_model.encoder.save_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_decoder_{self.checkpointname}')
            else:
                self.AE_model.save_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}{self.checkpointname}')
        
            print("Model weights saved")
            end1 = time.time()
            print(f"Time taken for megaset {megaset} is: {(end1-start1)/60:.2f}m or {(end1-start1)/60/60:.2f}h")
            print(" ")
            
            
            
        end = time.time()
        print(f"Time taken for all megasets is: {(end-start)/60:.2f}m or {(end-start)/60/60:.2f}h")
        print(" ")  
          
            
    def RunInference(self):
        
        
        try:
            self.AE_model
        except:
            #* Load model 
            if TYPE == "VAE":
                nn_model = RunVAE(data_shape=data_shape)
                if SMALL:
                    self.AE_model = nn_model.getModel()
                    
                else:
                    self.AE_model = nn_model.getModelBig()
                   
            elif TYPE == "AE":
                nn_model = RunAE(data_shape=data_shape)
                
                if SMALL:
                    self.AE_model = nn_model.getModel()
                    
                else:
                    self.AE_model = nn_model.getModelBig()
                    
                
            
        if TYPE == "VAE":
            self.AE_model.encoder.load_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_encoder_{self.checkpointname}')
            self.AE_model.encoder.load_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}_decoder_{self.checkpointname}')
        else:
            self.AE_model.load_weights(f'./checkpoints/Megabatch_checkpoint_{TYPE}{self.checkpointname}')

        #* Iterate over the different signals, by index
        
        val_cats = []
        val_weights = []
        recon_err = []
        
        xvals = []
        
        #* Validation inference for all megasets
        for megaset in range(self.totmegasets):
            print(f"Running inference on megabatch: {megaset}")
            
            FETCH_PATH = DATA_PATH/ f"Megabatches/MB{megaset}"
            MERGE_PATH = FETCH_PATH / f"MergedMB{megaset}"
            
            xval = np.load(MERGE_PATH / f"Merged{megaset}_xval.npy")
            x_val_weights = np.load(MERGE_PATH / f"Merged{megaset}_weights_val.npy")
            x_val_cats = np.load(MERGE_PATH / f"Merged{megaset}_categories_val.npy")
            
            xvals.append(xval)
            
            recon_err_back = self._inference(xval, types="MC")
                
            val_cats.append(x_val_cats)
            val_weights.append(x_val_weights)
            recon_err.append(recon_err_back)
            
            
        xvals = np.concatenate(xvals, axis=0)

        recon_err = np.concatenate(recon_err, axis=0)
        val_weights = np.concatenate(val_weights, axis=0)
        val_cats = np.concatenate(val_cats, axis=0)
        
        pattern = re.compile(r'(\D+)\d*')
        val_cats = np.array([re.sub(pattern, r'\1', elem) if any(x in elem for x in ['Zmmjets', 'Zeejets']) else elem for elem in val_cats])

        #* Signal inference 
        for signal_num in [0,1]:  
            
            sigs = np.unique(self.signal_categories)
            sig = sigs[signal_num]
            signame = sig[21:-9]
            print(f"{signame} start")
            sig_idx = np.where(self.signal_categories==sig)
            signal_cats = self.signal_categories.to_numpy()[sig_idx]
            signal = self.signal[sig_idx]
            recon_err_sig = self._inference(signal, types="Signal")
            sig_weights = self.signal_weights.to_numpy()[sig_idx] 
            
            plothisto = PlotHistogram(STORE_IMG_PATH, recon_err, val_weights, val_cats, signal=recon_err_sig, signal_weights=sig_weights, signal_cats=signal_cats)
            plothisto.histogram(self.channels, sig_name=signame)
            
            
            #* ROC curve
            self._roc_curve(distribution_bkg=xvals[:, 0], weights_bkg=val_weights, distribution_sig=signal[:, 0], weights_sig=sig_weights, sig_name=signame, figname="$e_T^{miss}$")
            
            self._roc_curve(distribution_bkg=recon_err, weights_bkg=val_weights, distribution_sig=recon_err_sig, weights_sig=sig_weights, sig_name=signame, figname="Reconstruction_error")

            print(f"{signame} done!")
    
        
    def _roc_curve(self, distribution_bkg, weights_bkg, distribution_sig, weights_sig, sig_name, figname):
        bkg_dist = distribution_bkg.copy()
        bkg = np.zeros(len(distribution_bkg))
        
        sig_dist = distribution_sig.copy()
        sg = np.ones(len(distribution_sig))
        
        label = np.concatenate((bkg, sg))
        scores = np.concatenate((bkg_dist,sig_dist))
        
        scaleFactor = np.sum(weights_sig) / np.sum(weights_bkg)
        
        weights = np.concatenate((weights_bkg*scaleFactor, weights_sig))
        
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
        plt.title(rf"ROC curve of {figname} for SM bkg and " + f"SUSY{sig_name}", fontsize=25)
        plt.savefig(STORE_IMG_PATH + f"histo/{LEP}/{TYPE}/{arc}/{SCALER}/roc_curve_{figname}_{sig_name}.pdf")
        plt.close()
    
        
     
    
    def _trainloop(self, xtrain, xval, x_train_weights):
        """Sets up the training loop for a given megaset

        Args:
            xtrain (np.ndarray): Training set
            xval (np.ndarray): Validation set
            x_train_weights (pd.DataFrame): Weights for the trianing set
        """
        
        with tf.device("/GPU:0"):

            tf.config.optimizer.set_jit("autoclustering")

            if TYPE == "VAE":
                self.AE_model.fit(
                    xtrain,
                    epochs=self.epochs,
                    batch_size=self.b_size,
                    sample_weight=x_train_weights,
                )
            else:
                self.AE_model.fit(
                    xtrain,
                    xtrain,
                    epochs=self.epochs,
                    batch_size=self.b_size,
                    validation_data=(xval, xval),
                    sample_weight=x_train_weights,
                ) 
    
    def _inference(self, arr, types="MC"):
        """Sets up inference for a given megaset

        Args:
            arr (np.ndarray): Matrix containing the data to have inference on

        Returns:
            np.ndarray: Reconstruction loss of arr
        """
        
        
        if types == "MC":
            datatype = "Background"
        elif types == "Signal":
            datatype = "Signal"
        else:
            datatype = "ATLAS Data"
            
        with tf.device("/CPU:0"):
            if TYPE == "VAE":
                print(f"{datatype} started")
                z_m, z_var, z = self.AE_model.encoder.predict(arr, batch_size=self.b_size)
                sig_err = self.AE_model.decoder.predict(z)
                print(f"{datatype} predicted")
                recon_err_sig = self._reconstructionError(sig_err, arr)
                print(f"{datatype} done, lenght: {len(recon_err_sig)}")
            else:
                print(f"{datatype} started")
                sig_err = self.AE_model.predict(arr, batch_size=self.b_size)
                print(f"{datatype} predicted")
                recon_err_sig = self._reconstructionError(sig_err, arr)
                print(f"{datatype} done, lenght: {len(recon_err_sig)}")
        
        return recon_err_sig
    
    def _reconstructionError(self, pred: np.ndarray, real: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            pred (np.ndarray): Prediction from model
            real (np.ndarray): Truth array to compare

        Returns:
            np.ndarray: log10 of the error, each feature weighted the same
        """

        diff = pred - real
        err = np.power(diff, 2)
        err = np.sum(err, 1)
        err = np.log10(err)
        return err
        
if __name__ == "__main__":
    
    
    gpus = tf.config.experimental.list_physical_devices('GPU') 
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)

    
    
    global data_shape
    data_shape = 529
    
    L2 = LEP2ScaleAndPrep(DATA_PATH, True, SAVE_VAR, LOAD_VAR, lep=2, convert=True)
    #L2.convertParquet()
    #L2.createMCSubsamples()
    
    #L2.mergeMegaBatches()
    
    #L2.RunTraining()
    L2.RunInference()