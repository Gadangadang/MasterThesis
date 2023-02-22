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
    def __init__(self, data_structure:object, path: str)->None:
        """
        Class to run training, inference and plotting

        Args:
            data_structure (object): Object containing the training, validation and test set
            path (Path): Path to store information
        """
        self.path = path
        self.data_structure = data_structure
        self.X_train = self.data_structure.X_b_train
        self.X_val = self.data_structure.X_b_val
        self.signal = self.data_structure.signal
        self.signal_cats = self.data_structure.signal_categories
        self.signal_weights = self.data_structure.signal_weights
        self.data = self.data_structure.data
        self.data_shape = np.shape(self.X_train)[1]
        self.idxs = self.data_structure.idxs
        self.val_cats = self.data_structure.val_categories.to_numpy()
        self.err_val = self.data_structure.weights_val.to_numpy()
        self.err_train = self.data_structure.weights_train.to_numpy()
        
        self.X_train_trilep_mass = self.data_structure.X_train_trilep_mass.to_numpy()
        self.X_val_trilep_mass = self.data_structure.X_val_trilep_mass.to_numpy()
        self.data_trilep_mass = self.data_structure.data_trilep_mass.to_numpy()
        self.signal_trilep_mass = self.data_structure.signal_trilep_mass.to_numpy()
        
        self.X_train_eTmiss = self.data_structure.X_train_eTmiss.to_numpy()
        self.X_val_eTmiss = self.data_structure.X_val_eTmiss.to_numpy()
        self.data_eTmiss = self.data_structure.data_eTmiss.to_numpy()
        self.signal_eTmiss = self.data_structure.signal_eTmiss.to_numpy()

        print(" ")
        print(f"{(self.X_train.nbytes + self.X_val.nbytes)/1000000000} Gbytes")
        print(" ")

        self.sample_weight = self.data_structure.weights_train

        self.channels = [channel for channel, _, __ in self.idxs]

        self.name = "test"

        self.b_size = BACTH_SIZE

        self.epochs = EPOCHS

    
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
    
    
    def getModel_big(self):
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
        

    

    def trainModel(self, X_train: np.ndarray, X_val: np.ndarray, sample_weight: dict)->None:
        """_summary_

        Args:
            X_train (_type_): _description_
            X_val (_type_): _description_
            sample_weight (_type_): _description_
        """

    
        if SMALL:
            self.AE_model = self.getModel()
            print("New model created")
        else:
            self.AE_model = self.getModel_big()
            print("New model created")

        #print(self.AE_model.layers[1].weights)
        with tf.device("/GPU:0"):

            tf.config.optimizer.set_jit("autoclustering")

            self.AE_model.fit(
                X_train,
                epochs=self.epochs,
                batch_size=self.b_size,
                sample_weight=sample_weight,
            )

            print("Fitting complete")

        self.modelname = f"model_{self.name}"
        #self.AE_model.save("tf_models/" + self.modelname + ".h5")

        #print(f"{self.modelname} saved")


      

    def runInference(self, X_val: np.ndarray, test_set: np.ndarray, tuned_model=False)->None:
        """_summary_

        Args:
            X_val (np.ndarray): _description_
            test_set (np.ndarray): _description_
            tuned_model (bool, optional): _description_. Defaults to False.
        """
        try:
            self.AE_model
            print("Model loaded")
        except:
            if tuned_model:
                print("tuned trained_model")
                try:
                    self.modelname
                except:
                    self.modelname = input("Modelname: ")
                    ending = self.modelname.find(".h5")
                    if ending > -1:
                        self.modelname = self.modelname[:ending]

                self.AE_model = tf.keras.models.load_model(
                    "tf_models/" + self.modelname + ".h5"
                )
            else:
                print("reg trained_model")
                self.AE_model = self.trainModel()

        
        with tf.device("/CPU:0"):
            print("Background started")
            z_m, z_var, z = self.AE_model.encoder.predict(X_val, batch_size=self.b_size)
            self.pred_back = self.AE_model.decoder.predict(z)
            print("Background predicted")
            print(np.shape(self.pred_back))
            self.recon_err_back = self.reconstructionError(self.pred_back, X_val)
            
            
            print(f"Background done, lenght: {len(self.recon_err_back)}")

            if len(test_set) > 0:
                print("Signal started")
                z_m, z_var, z = self.AE_model.encoder.predict(test_set, batch_size=self.b_size)
                self.pred_sig = self.AE_model.decoder.predict(z)
                self.recon_sig = self.reconstructionError(self.pred_sig, test_set)
                print(f"Signal done, lenght: {len(self.recon_sig)}")

            print("ATLAS data started")
            z_m, z_var, z = self.AE_model.encoder.predict(self.data, batch_size=self.b_size)
            self.pred_data = self.AE_model.decoder.predict(z)
           
            
            self.recon_data = self.reconstructionError(self.pred_data, self.data)
            print("ATLAS data done")

    def reconstructionError(self, pred: np.ndarray, real: np.ndarray) -> np.ndarray:
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

    def checkReconError(self, channels: list, sig_name="nosig", Noise=False)->None:
        """_summary_

        Args:
            channels (list): List containing all the channels
            sig_name (str, optional): Name of signal sample. Defaults to "nosig".
            Noise (bool, optional): Noise paramter, sets the bins to 25 as default. Defaults to False.
        """

        histo_atlas = []
        weight_atlas_data = []
        try:
    
            for id, channel in enumerate(channels):
                
                
                idxs = self.tot_weights_per_channel[id]
                print(id, len(idxs), len(self.recon_err_back))
                err = self.recon_err_back[idxs]

                histo_atlas.append(err)

                err_w = self.err_val[idxs]

                weight_atlas_data.append(err_w)
        except:
            for channel in channels:

                err = self.recon_err_back[np.where(self.val_cats == channel)[0]]

                histo_atlas.append(err)

                err_w = self.err_val[np.where(self.val_cats == channel)[0]]

                weight_atlas_data.append(err_w)
            
            
        try:
            sig_err = self.recon_sig
            sig_err_w = self.sig_err
        except:
            print("No signal")
            
            
        
        

        sum_w = [np.sum(weight) for weight in weight_atlas_data]
        sort_w = np.argsort(sum_w, kind="mergesort")

        sns.set_style("darkgrid")
        plt.rcParams["figure.figsize"] = (12, 9)

        fig, ax = plt.subplots()

        try:
            print("in try 1")
            N, bins = np.histogram(sig_err, bins=25, weights=sig_err_w)
            print("in try 2")
            x = (np.array(bins[0:-1]) + np.array(bins[1:])) / 2
            print("in try 3")
            ax.scatter(x, N, marker="+", label=f"{sig_name}", color="black")  # type: ignore
            print("in try 4")
            n_bins = bins
            print("Bins: ",n_bins)
        except:
            n_bins = 25
    
        if Noise:
            n_bins = 25

        colors = [
            "mediumspringgreen",
            "darkgreen",
            "lime",
            "magenta",
            "blue",
            "red",
            "orange",
            "brown",
            "cyan",
            "mediumorchid",
            "gold",
            "darkgoldenrod",
            
        ]
        
        if len(colors) != len(histo_atlas):
            colors = np.random.choice(colors, size=len(histo_atlas), replace=False)  
        
        if len(histo_atlas) < 2:
            channels = ["Monte Carlo"]
            
        self.n_bins = n_bins
            
        print(colors, len(histo_atlas), channels)
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
            n_bins,
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
                "Reconstruction error histogram with MC and ATLAS data", fontsize=25
            )
        else:
            ax.set_title(
                "Reconstruction error histogram with MC", fontsize=25
            )
        ax.set_xlabel("Log10 Reconstruction Error", fontsize=25)
        ax.set_ylabel("#Events", fontsize=25)
        # ax.set_xlim([0, 3.5])
        ax.set_ylim([0.1, 5e6])  # type: ignore
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=25)
        fig.tight_layout()
        
        plt.savefig(self.path + f"histo/{LEP}/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{sig_name}.pdf")
        plt.close()



