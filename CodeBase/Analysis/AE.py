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



class RunAE:
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
        # Input layer
        inputs = tf.keras.layers.Input(shape=self.data_shape, name="encoder_input")

        # First hidden layer
        x = tf.keras.layers.Dense(
            units=70,
            activation="tanh",
            kernel_regularizer=tf.keras.regularizers.L1(0.05),
            activity_regularizer=tf.keras.regularizers.L2(0.5),
        )(inputs)

        # Second hidden layer
        x_ = tf.keras.layers.Dense(units=45, activation="linear")(inputs)

        # Third hidden layer
        x1 = tf.keras.layers.Dense(
            units=20,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.L1(0.05),
            activity_regularizer=tf.keras.regularizers.L2(0.5),
        )(x_)

        val = 7

        # Forth hidden layer
        x2 = tf.keras.layers.Dense(
            units=val, activation=tf.keras.layers.LeakyReLU(alpha=1)
        )(x1)

        # Encoder definition
        encoder = tf.keras.Model(inputs, x2, name="encoder")
        encoder.summary()

        # Latent space
        latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")

        # Fifth hidden layer
        x = tf.keras.layers.Dense(
            units=22,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1(0.05),
            activity_regularizer=tf.keras.regularizers.L2(0.5),
        )(latent_input)

        # Sixth hidden layer
        x_ = tf.keras.layers.Dense(
            units=50, activation=tf.keras.layers.LeakyReLU(alpha=1)
        )(x)

        # Seventh hidden layer
        x1 = tf.keras.layers.Dense(
            units=73,
            activation="tanh",
            kernel_regularizer=tf.keras.regularizers.L1(0.05),
            activity_regularizer=tf.keras.regularizers.L2(0.5),
        )(x_)

        # Output layer
        output = tf.keras.layers.Dense(self.data_shape, activation="linear")(x1)

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

        # tf.keras.utils.plot_model(AE_model, to_file=path+"ae_model_plot.pdf", show_shapes=True, show_layer_names=True, expand_nested=True)

        return AE_model

    def trainModel(self, X_train: np.ndarray, X_val: np.ndarray, sample_weight: dict)->None:
        """_summary_

        Args:
            X_train (_type_): _description_
            X_val (_type_): _description_
            sample_weight (_type_): _description_
        """

        try:
            self.AE_model
            print("Model loaded")
        except:
            self.AE_model = self.getModel()
            print("New model created")

        #print(self.AE_model.layers[1].weights)
        with tf.device("/GPU:0"):

            tf.config.optimizer.set_jit("autoclustering")

            self.AE_model.fit(
                X_train,
                X_train,
                epochs=self.epochs,
                batch_size=self.b_size,
                validation_data=(X_val, X_val),
                sample_weight=sample_weight,
            )

            print("Fitting complete")

        self.modelname = f"model_{self.name}"
        self.AE_model.save("tf_models/" + self.modelname + ".h5")

        print(f"{self.modelname} saved")


      

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
            self.pred_back = self.AE_model.predict(X_val, batch_size=self.b_size)
            print("Background predicted")
            self.recon_err_back = self.reconstructionError(self.pred_back, X_val)
            print(f"Background done, lenght: {len(self.recon_err_back)}")

            if len(test_set) > 0:
                print("Signal started")
                self.pred_sig = self.AE_model.predict(test_set, batch_size=self.b_size)
                self.recon_sig = self.reconstructionError(self.pred_sig, test_set)
                print(f"Signal done, lenght: {len(self.recon_sig)}")

            print("ATLAS data started")
            self.pred_data = self.AE_model.predict(self.data, batch_size=self.b_size)
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
            N, bins = np.histogram(sig_err, bins=25, weights=sig_err_w)
            x = (np.array(bins[0:-1]) + np.array(bins[1:])) / 2
            ax.scatter(x, N, marker="+", label=f"{sig_name}", color="black")  # type: ignore
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
        
        plt.savefig(self.path + f"histo/{TYPE}/{arc}/{SCALER}/b_data_recon_big_rm3_feats_sig_{sig_name}.pdf")
        plt.close()


