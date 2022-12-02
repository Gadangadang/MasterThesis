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





class HyperParameterTuning(RunAE):
    def __init__(self, data_structure: object, path: str)->None:
        super().__init__(data_structure, path)
        
    def runHpSearch(
        self, X_train: np.ndarray, X_val: np.ndarray, sample_weight: dict, small=False, epochs=20
    )->None:
        """_summary_"""

        print(small)
        device_lib.list_local_devices()
        tf.config.optimizer.set_jit("autoclustering")
        with tf.device("/GPU:0"):
            if small:
                print("Small network enabled")
                self.gridautoencoder_small(X_train, X_val, sample_weight, epochs=epochs)
            else:
                self.gridautoencoder(X_train, X_val, sample_weight, epochs=epochs)

    def gridautoencoder(
        self, X_b: np.ndarray, X_back_test: np.ndarray, sample_weight: dict, epochs=20
    ) -> None:
        """_summary_

        Args:
            X_b (np.ndarray): _description_
            X_back_test (np.ndarray): _description_
        """
        tuner = kt.Hyperband(
            self.AE_model_builder,
            objective=kt.Objective("val_mse", direction="min"),
            max_epochs=epochs,
            factor=3,
            directory="GridSearches",
            project_name="AE",
            overwrite=True,
        )
        print(tuner.search_space_summary())

        tuner.search(
            X_b,
            X_b,
            epochs=epochs,
            batch_size=self.b_size,
            validation_data=(X_back_test, X_back_test),
            sample_weight=sample_weight,
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(
            f"""
        For Encoder: \n 
        First layer has {best_hps.get('num_of_neurons1')} with activation {best_hps.get('1_act')} \n
        Second layer has {best_hps.get('num_of_neurons2')} with activation {best_hps.get('2_act')} \n
        Third layer has {best_hps.get('num_of_neurons3')} with activation {best_hps.get('3_act')} \n
        
        Latent layer has {best_hps.get("lat_num")} with activation {best_hps.get('2_act')} \n
        \n
        For Decoder: \n 
        First layer has {best_hps.get('num_of_neurons5')} with activation {best_hps.get('5_act')}\n
        Second layer has {best_hps.get('num_of_neurons6')} with activation {best_hps.get('6_act')}\n
        Third layer has {best_hps.get('num_of_neurons7')} with activation {best_hps.get('7_act')}\n
        Output layer has activation {best_hps.get('8_act')}\n
        \n
        with learning rate = {best_hps.get('learning_rate')} and alpha = {best_hps.get('alpha')}
        """
        )

        state = True
        while state == True:
            # answ = input("Do you want to save model? (y/n) ")
            # if answ == "y":
            # name = input("name: model_ ")

            self.AE_model = tuner.hypermodel.build(best_hps)
            self.modelname = f"model_{self.name}"
            self.AE_model.save("tf_models/" + self.modelname + ".h5")
            state = False
            print(f"Model {self.modelname} saved")

            #
            """
            elif answ == "n":
                state = False
                print("Model not saved")
            """

    def gridautoencoder_small(
        self, X_b: np.ndarray, X_back_test: np.ndarray, sample_weight: dict, epochs=20
    ) -> None:
        """_summary_

        Args:
            X_b (np.ndarray): _description_
            X_back_test (np.ndarray): _description_
        """
        tuner = kt.Hyperband(
            self.AE_model_builder_small,
            objective=kt.Objective("val_mse", direction="min"),
            max_epochs=1,#epochs,
            factor=3,
            directory="GridSearches",
            project_name="AE",
            overwrite=True,
        )
        print(tuner.search_space_summary())

        tuner.search(
            X_b,
            X_b,
            epochs=1,#epochs,
            batch_size=self.b_size,
            validation_data=(X_back_test, X_back_test),
            sample_weight=sample_weight,
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        #print(tuner.search_space_summary())

        state = True
        while state == True:
            # answ = input("Do you want to save model? (y/n) ")
            # if answ == "y":
            # name = input("name: model_ ")
            self.AE_model = tuner.hypermodel.build(best_hps)
            self.modelname = f"model_{self.name}"
            self.AE_model.save("tf_models/" + self.modelname + ".h5")
            state = False
            print(f"Model {self.modelname} saved")

            #
            """
            elif answ == "n":
                state = False
                print("Model not saved")
            """

    def AE_model_builder(self, hp: kt.engine.hyperparameters.HyperParameters):

        """_summary_

        Args:
            hp (kt.engine.hyperparameters.HyperParameters): _description_

        Returns:
            _type_: _description_
        """
        ker_choice = hp.Choice("Kernel_reg", values=[0.5, 0.1, 0.05, 0.01])
        act_choice = hp.Choice("Atc_reg", values=[0.5, 0.1, 0.05, 0.01])

        alpha_choice = hp.Choice("alpha", values=[1.0, 0.5, 0.1, 0.05, 0.01])

        # Activation functions
        activations = {
            "relu": tf.nn.relu,
            "tanh": tf.nn.tanh,
            "leakyrelu": "leaky_relu",
            "linear": tf.keras.activations.linear,
        }  # lambda x: tf.nn.leaky_relu(x, alpha=alpha_choice),

        # Input layer
        inputs = tf.keras.layers.Input(shape=self.data_shape, name="encoder_input")

        # First hidden layer
        x = tf.keras.layers.Dense(
            units=hp.Int(
                "num_of_neurons1", min_value=60, max_value=self.data_shape - 1, step=1
            ),
            activation=activations.get(
                hp.Choice("1_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(inputs)

        # Second hidden layer
        x_ = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons2", min_value=30, max_value=59, step=1),
            activation=activations.get(
                hp.Choice("2_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x)

        # Third hidden layer
        x1 = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons3", min_value=10, max_value=29, step=1),
            activation=activations.get(
                hp.Choice("3_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(x_)

        val = hp.Int("lat_num", min_value=1, max_value=9, step=1)

        # Forth hidden layer
        x2 = tf.keras.layers.Dense(
            units=val,
            activation=activations.get(
                hp.Choice("4_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x1)

        # Encoder definition
        encoder = tf.keras.Model(inputs, x2, name="encoder")

        # Latent space
        latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")

        # Fifth hidden layer
        x = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons5", min_value=10, max_value=29, step=1),
            activation=activations.get(
                hp.Choice("5_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(latent_input)

        # Sixth hidden layer
        x_ = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons6", min_value=30, max_value=59, step=1),
            activation=activations.get(
                hp.Choice("6_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x)

        # Seventh hidden layer
        x1 = tf.keras.layers.Dense(
            units=hp.Int(
                "num_of_neurons7", min_value=60, max_value=self.data_shape - 1, step=1
            ),
            activation=activations.get(
                hp.Choice("7_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(x_)

        # Output layer
        output = tf.keras.layers.Dense(
            self.data_shape,
            activation=activations.get(
                hp.Choice("8_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x1)

        # Encoder definition
        decoder = tf.keras.Model(latent_input, output, name="decoder")

        # Output definition
        outputs = decoder(encoder(inputs))

        # Model definition
        AE_model = tf.keras.Model(inputs, outputs, name="AE_model")

        hp_learning_rate = hp.Choice(
            "learning_rate", values=[9e-2, 9.5e-2, 1e-3, 1.5e-3]
        )
        optimizer = tf.keras.optimizers.Adam(hp_learning_rate)

        AE_model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

        return AE_model

    def AE_model_builder_small(self, hp: kt.engine.hyperparameters.HyperParameters):
        """_summary_
        Args:
            hp (kt.engine.hyperparameters.HyperParameters): _description_
        Returns:
            _type_: _description_
        """
        ker_choice = hp.Choice("Kernel_reg", values=[0.5, 0.1, 0.05, 0.01])
        act_choice = hp.Choice("Atc_reg", values=[0.5, 0.1, 0.05, 0.01])

        alpha_choice = hp.Choice("alpha", values=[1.0, 0.5, 0.1, 0.05, 0.01])

        # Activation functions
        activations = {
            "relu": tf.nn.relu,
            "tanh": tf.nn.tanh,
            "leakyrelu": "leaky_relu",
            "linear": tf.keras.activations.linear,
        }  # lambda x: tf.nn.leaky_relu(x, alpha=alpha_choice),

        # Input layer
        inputs = tf.keras.layers.Input(shape=self.data_shape, name="encoder_input")

        # Third hidden layer
        x1 = tf.keras.layers.Dense(
            units=self.data_shape,#hp.Int("num_of_neurons1", min_value=20, max_value=self.data_shape, step=5),
            activation=activations.get(
                hp.Choice("1_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(inputs)
        
        drop = tf.keras.layers.Dropout(.2)(x1)
        
        val = hp.Int("lat_num", min_value=2, max_value=int(self.data_shape/2), step=3)

        # Forth hidden layer
        x2 = tf.keras.layers.Dense(
            units=val,
            activation=activations.get(
                hp.Choice("2_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(drop)

        """# Encoder definition
        encoder = tf.keras.Model(inputs, x2, name="encoder")

        # Latent space
        latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")"""

        # Output layer
        output = tf.keras.layers.Dense(
            self.data_shape,
            activation=activations.get(
                hp.Choice("3_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x2)

        """# Encoder definition
        decoder = tf.keras.Model(latent_input, output, name="decoder")
        """
        # Output definition
        #outputs = output(encoder(inputs))

        # Model definition
        AE_model = tf.keras.Model(inputs, output, name="AE_model")

        hp_learning_rate = hp.Choice(
            "learning_rate", values=[9e-2, 9.5e-2, 1e-3, 1.5e-3]
        )
        optimizer = tf.keras.optimizers.Adam(hp_learning_rate)

        AE_model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

        return AE_model

