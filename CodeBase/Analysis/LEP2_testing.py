import re
import os
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
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction))
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
        x = tf.keras.layers.Dense(units=self.data_shape, activation="relu")(
            encoder_inputs
        )
        x = tf.keras.layers.Dense(
            units=400,
            activation="tanh",
        )(encoder_inputs)
        x = tf.keras.layers.Dense(units=300, activation="relu")(x)
        x = tf.keras.layers.Dense(
            200,
            activation=tf.keras.layers.LeakyReLU(alpha=0.3),
        )(x)

        z_mean = tf.keras.layers.Dense(val, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(val, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        latent_inputs = tf.keras.Input(shape=val)
        x = tf.keras.layers.Dense(
            units=200, activation=tf.keras.layers.LeakyReLU(alpha=0.3)
        )(latent_inputs)
        x = tf.keras.layers.Dense(
            units=300,
            activation="relu",
        )(x)
        x = tf.keras.layers.Dense(
            units=400,
            activation="tanh",
        )(x)
        decoder_outputs = tf.keras.layers.Dense(
            units=self.data_shape, activation="sigmoid"
        )(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        self.AE_model = VAE(encoder, decoder)
        self.AE_model.compile(optimizer=tf.keras.optimizers.Adam())
        # self.AE_model.save("tf_models/untrained_big_vae")
        return self.AE_model

    def getModel(self):
        """_summary_

        Returns:
            tf.python.keras.engine.functional.Functional: Model to use
        """

        val = 150

        encoder_inputs = tf.keras.Input(shape=self.data_shape)
        x = tf.keras.layers.Dense(units=self.data_shape, activation="relu")(
            encoder_inputs
        )

        z_mean = tf.keras.layers.Dense(val, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(val, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        latent_inputs = tf.keras.Input(shape=val)

        decoder_outputs = tf.keras.layers.Dense(
            units=self.data_shape, activation="sigmoid"
        )(latent_inputs)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        self.AE_model = VAE(encoder, decoder)
        self.AE_model.compile(optimizer=tf.keras.optimizers.Adam())
        # self.AE_model.save("tf_models/untrained_small_vae")

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
        x2 = tf.keras.layers.Dense(
            units=val, activation=tf.keras.layers.LeakyReLU(alpha=0.3)
        )(x)

        # Encoder definition
        encoder = tf.keras.Model(inputs, x2, name="encoder")
        encoder.summary()

        # Latent input for decoder
        latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")

        # Output layer
        output = tf.keras.layers.Dense(
            self.data_shape, activation=tf.keras.layers.LeakyReLU(alpha=0.3)
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
        # AE_model.save("tf_models/untrained_small_ae")

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
        x_ = tf.keras.layers.Dense(
            units=300, activation=tf.keras.layers.LeakyReLU(alpha=0.3)
        )(x__)

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
        output = tf.keras.layers.Dense(
            self.data_shape, activation=tf.keras.layers.LeakyReLU(alpha=0.3)
        )(x1)

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
        # AE_model.save("tf_models/untrained_big_ae")
        # tf.keras.utils.plot_model(AE_model, to_file=path+"ae_model_plot.pdf", show_shapes=True, show_layer_names=True, expand_nested=True)

        return AE_model


class LEP2ScaleAndPrep:
    def __init__(
        self, path: Path, event_rmm=False, save=False, load=False, lep=3, convert=True
    ) -> None:
        """_summary_

        Args:
            path (Path): _description_
            event_rmm (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
            load (bool, optional): _description_. Defaults to False.
            lep (int, optional): _description_. Defaults to 3.
            convert (bool, optional): _description_. Defaults to True.
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
            "Diboson",
        ]

        self.parqs = [
            f
            for f in listdir(self.path)
            if isfile(join(self.path, f)) and f[-8:] == ".parquet"
        ]

        self.signal = np.load(DATA_PATH / "signal.npy")
        self.signal_weights = pd.read_hdf(DATA_PATH / "signal_weight_b.h5")
        self.signal_categories = pd.read_hdf(DATA_PATH / "signal_cat_b.h5")
        self.signal_etmiss = pd.read_hdf(DATA_PATH / "signal_etmiss.h5").to_numpy()

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
            FETCH_PATH = DATA_PATH / f"Megabatches/MB{megaset}"
            MERGE_PATH = FETCH_PATH / f"MergedMB{megaset}"

            xtrain = np.load(MERGE_PATH / f"Merged{megaset}_xtrain.npy")
            xval = np.load(MERGE_PATH / f"Merged{megaset}_xval.npy")
            x_train_weights = pd.DataFrame(
                np.load(MERGE_PATH / f"Merged{megaset}_weights_train.npy")
            )
            
            
            # * Load model
            if SMALL:

                self.AE_model = nn_model.getModel()
            else:

                self.AE_model = nn_model.getModelBig()

            file = f"timelist_model_{TYPE}_{size_m}.txt"

            if megaset != 0:
                if TYPE == "VAE":
                    self.AE_model.encoder.load_weights(
                        f"./checkpoints/Megabatch_checkpoint_{TYPE}_encoder_{self.checkpointname}"
                    )
                    self.AE_model.encoder.load_weights(
                        f"./checkpoints/Megabatch_checkpoint_{TYPE}_decoder_{self.checkpointname}"
                    )
                else:

                    self.AE_model.load_weights(
                        f"./checkpoints/Megabatch_checkpoint_{TYPE}{self.checkpointname}"
                    )

            # * Run Training
            self._trainloop(xtrain, xval, x_train_weights)

            if TYPE == "VAE":
                self.AE_model.encoder.save_weights(
                    f"./checkpoints/Megabatch_checkpoint_{TYPE}_encoder_{self.checkpointname}"
                )
                self.AE_model.encoder.save_weights(
                    f"./checkpoints/Megabatch_checkpoint_{TYPE}_decoder_{self.checkpointname}"
                )
            else:
                self.AE_model.save_weights(
                    f"./checkpoints/Megabatch_checkpoint_{TYPE}{self.checkpointname}"
                )

            print("Model weights saved")
            end1 = time.time()
            megasettime = end1 - start1
            print(
                f"Time taken for megaset {megaset} is: {(megasettime)/60:.2f}m or {(megasettime)/60/60:.2f}h"
            )
            print(" ")

            

        end = time.time()
        print(
            f"Time taken for all megasets is: {(end-start)/60:.2f}m or {(end-start)/60/60:.2f}h"
        )
        print(" ")

        

    def RunInference(self):

        try:
            self.AE_model
        except:
            # * Load model
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
            self.AE_model.encoder.load_weights(
                f"./checkpoints/Megabatch_checkpoint_{TYPE}_encoder_{self.checkpointname}"
            )
            self.AE_model.encoder.load_weights(
                f"./checkpoints/Megabatch_checkpoint_{TYPE}_decoder_{self.checkpointname}"
            )
        else:
            self.AE_model.load_weights(
                f"./checkpoints/Megabatch_checkpoint_{TYPE}{self.checkpointname}"
            )

        # * Iterate over the different signals, by index

        file = f"significance_{TYPE}_{size_m}.txt"

        val_cats = []
        val_weights = []
        recon_err = []
        etmiss = []
        xvals = []
        
        train_weights = []

        # * Validation inference for all megasets
        for megaset in range(self.totmegasets):
            print(f"Running inference on megabatch: {megaset}")

            FETCH_PATH = DATA_PATH / f"Megabatches/MB{megaset}"
            MERGE_PATH = FETCH_PATH / f"MergedMB{megaset}"

            xval = np.load(MERGE_PATH / f"Merged{megaset}_xval.npy")
            x_val_weights = np.load(MERGE_PATH / f"Merged{megaset}_weights_val.npy")
            x_train_weights = np.load(MERGE_PATH / f"Merged{megaset}_weights_train.npy")
            x_val_cats = np.load(MERGE_PATH / f"Merged{megaset}_categories_val.npy")
            etmiss_mega = np.load(MERGE_PATH / f"Merged{megaset}_etmiss_val.npy")

            print("etmiss stuff: ")
            print("vals: ", etmiss_mega)
            print("max: ", np.max(etmiss_mega))
            print("mean: ", np.mean(etmiss_mega))
            print("median: ", np.median(etmiss_mega))

            xvals.append(xval)
            etmiss.append(etmiss_mega)
            recon_err_back = self._inference(xval, types="MC")

            train_weights.append(x_train_weights)
            val_cats.append(x_val_cats)
            val_weights.append(x_val_weights)
            recon_err.append(recon_err_back)

        etmiss = np.concatenate(etmiss, axis=0)
        etmiss_cut = np.where(etmiss < 800)
        print("Etmiss cut to remove all above 800 GeV")
        print(np.shape(etmiss_cut))
        print(" ")
        etmiss = etmiss[etmiss_cut]
        xvals = np.concatenate(xvals, axis=0)[etmiss_cut]
        recon_err = np.concatenate(recon_err, axis=0)[etmiss_cut]
        val_we = np.concatenate(val_weights, axis=0)
        val_weights = val_we[etmiss_cut]
        
        train_weights = np.concatenate(train_weights, axis=0)
        
        print(f"Sum events: {np.sum(train_weights) + np.sum(val_we)}")
      
        
        val_cats = np.concatenate(val_cats, axis=0)[etmiss_cut]

        pattern = re.compile(r"(\D+)\d*")
        val_cats = np.array(
            [
                re.sub(pattern, r"\1", elem)
                if any(x in elem for x in ["Zmmjets", "Zeejets"])
                else elem
                for elem in val_cats
            ]
        )

        # * Signal inference
        for signal_num in [0, 1]:

            sigs = np.unique(self.signal_categories)
            sig = sigs[signal_num]
            signame = sig[21:-9]
            string_write = f"\nSignal: {signame}\n"
            write_to_file(file, string_write)
            print(f"{signame} start")
            sig_idx = np.where(self.signal_categories == sig)
            signal_cats = self.signal_categories.to_numpy()[sig_idx]
            signal = self.signal[sig_idx]
            recon_err_sig = self._inference(signal, types="Signal")
            sig_weights = self.signal_weights.to_numpy()[sig_idx]

            plothisto = PlotHistogram(
                STORE_IMG_PATH,
                recon_err,
                val_weights,
                val_cats,
                signal=recon_err_sig,
                signal_weights=sig_weights,
                signal_cats=signal_cats,
            )
            plothisto.histogram(self.channels, sig_name=signame)

            # * ROC curve
            """self._roc_curve(distribution_bkg=xvals[:, 0], 
                            weights_bkg=val_weights, 
                            distribution_sig=signal[:, 0], 
                            weights_sig=sig_weights, 
                            sig_name=signame, 
                            figname="$e_T^{miss}$")"""

            """self._roc_curve(distribution_bkg=recon_err, 
                            weights_bkg=val_weights, 
                            distribution_sig=recon_err_sig, 
                            weights_sig=sig_weights, 
                            sig_name=signame, 
                            figname="Reconstruction_error")"""

            # * Etmiss pre cut
            histo_tit = r"$e_T^{miss}$ distribution for SM MC and " + f"{signame}"
            plotetmiss = PlotHistogram(
                STORE_IMG_PATH,
                etmiss,
                val_weights,
                val_cats,
                histoname=histo_tit,
                featurename=r"$e_T^{miss}$",
                histotitle=histo_tit,
                signal=self.signal_etmiss[sig_idx],
                signal_weights=sig_weights,
                signal_cats=signal_cats,
            )
            plotetmiss.histogram(self.channels, sig_name=signame, etmiss_flag=True)

            small_sig = _significance_small(np.sum(sig_weights), np.sum(val_weights))
            big_sig = _significance_big(np.sum(sig_weights), np.sum(val_weights))

            print(" ")
            print(
                f"Pre cut etmiss;  Signifance small: {small_sig} | Significance big: {big_sig}"
            )
            print(" ")

            string_write = f"\nPre reconstruction error cut:\n"
            write_to_file(file, string_write)

            string_write = (
                f"Significance small: {small_sig} | Signifiance big: {big_sig}\n"
            )
            write_to_file(file, string_write)

            # * Etmiss post Reconstruction cut
            median = np.median(plothisto.n_bins)
            std = np.abs(median / 5)
            print(f"Median recon: {median}, std recon: {std}")

            string_write = f"\nPost recon err cut\n"
            write_to_file(file, string_write)

            for std_scale in range(1, 4):

                recon_er_cut = median + std_scale * std
                print(f"Recon err cut: {recon_er_cut}")

                error_cut_val = np.where(recon_err > (recon_er_cut))[0]
                error_cut_sig = np.where(recon_err_sig > (recon_er_cut))[0]

                print(f"val cut shape: {np.shape(error_cut_val)}")

                val_weights_cut = val_weights[error_cut_val]
                sig_weights_cut = sig_weights[error_cut_sig]

                val_cats_cut = val_cats[error_cut_val]
                signal_cats_cut = signal_cats[error_cut_sig]

                etmiss_bkg = etmiss[error_cut_val]
                etmiss_sig = self.signal_etmiss[error_cut_sig]

                small_sig = _significance_small(
                    np.sum(sig_weights_cut), np.sum(val_weights_cut)
                )
                big_sig = _significance_big(
                    np.sum(sig_weights_cut), np.sum(val_weights_cut)
                )

                print(" ")
                print(
                    f"S: {np.sum(sig_weights_cut):.3f} | B: {np.sum(val_weights_cut):.3f}"
                )
                print(f"Signifance small: {small_sig} | Significance big: {big_sig}")
                print(" ")

                string_write = f"Recon error cut: {recon_er_cut}\n"
                write_to_file(file, string_write)

                string_write = (
                    f"Significance small: {small_sig} | Signifiance big: {big_sig}\n"
                )
                write_to_file(file, string_write)

                etmiss_histoname = (
                    r"$e_T^{miss}$ with recon err cut of " + f"{recon_er_cut:.2f}"
                )

                plotetmiss_cut = PlotHistogram(
                    STORE_IMG_PATH,
                    etmiss_bkg,
                    val_weights_cut,
                    val_cats_cut,
                    histoname=etmiss_histoname,
                    featurename=r"$e_T^{miss}$",
                    histotitle=f"recon_errcut_{recon_er_cut:.2f}",
                    signal=etmiss_sig,
                    signal_weights=sig_weights_cut,
                    signal_cats=signal_cats_cut,
                )
                plotetmiss_cut.histogram(
                    self.channels, sig_name=signame, etmiss_flag=True
                )

                if not isinstance(plotetmiss_cut.n_bins, int):

                    small_sign, big_sign = bin_integrate_significance(
                        plotetmiss_cut.n_bins,
                        etmiss_bkg,
                        etmiss_sig,
                        val_weights_cut,
                        sig_weights_cut,
                    )

                    plt.plot(
                        plotetmiss_cut.n_bins,
                        small_sign,
                        "r-",
                        label=r"$\sqrt{2((s+b)log(1+\frac{s}{b}) - s)}$",
                    )
                    plt.plot(
                        plotetmiss_cut.n_bins,
                        big_sign,
                        "b-",
                        label=r"$\frac{s}{\sqrt{b}}$",
                    )
                    plt.legend()
                    plt.xlabel(r"$e_T^{miss}$ [GeV]", fontsize=25)
                    plt.ylabel("Signifiance", fontsize=25)
                    plt.legend(prop={"size": 15})
                    plt.title(r"Significance as function of $e_T^{miss}$", fontsize=25)
                    plt.savefig(
                        STORE_IMG_PATH
                        + f"histo/{LEP}/{TYPE}/{arc}/{SCALER}/significance_etmiss_{signame}_{recon_er_cut}.pdf"
                    )
                    plt.close()

            print(f"{signame} done!")
            
    def checkIsBSM(self):
        
        
        try:
            self.AE_model
        except:
            # * Load model
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
            self.AE_model.encoder.load_weights(
                f"./checkpoints/Megabatch_checkpoint_{TYPE}_encoder_{self.checkpointname}"
            )
            self.AE_model.encoder.load_weights(
                f"./checkpoints/Megabatch_checkpoint_{TYPE}_decoder_{self.checkpointname}"
            )
        else:
            self.AE_model.load_weights(
                f"./checkpoints/Megabatch_checkpoint_{TYPE}{self.checkpointname}"
            )
            
        
        
        

        isBSM = []
        
        # * Validation inference for all megasets
        for megaset in range(self.totmegasets):
            print(f"Running inference on megabatch: {megaset}")

            FETCH_PATH = DATA_PATH / f"Megabatches/MB{megaset}"
            MERGE_PATH = FETCH_PATH / f"MergedMB{megaset}"

            #* Data 15 and data 16
            
            
            isBSM_test = np.load(MERGE_PATH / f"Merged{megaset}_data1516_isBSM.npy")
            
            isBSM.append(isBSM_test)
            
            if megaset == 3:
                break
        
        isBSM = np.concatenate(isBSM, axis=0) 
        print(np.unique(isBSM))
    
    def RunBlindTest(self, run_inference=True, create_rmm=True):
        
        
        try:
            self.AE_model
        except:
            # * Load model
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
            self.AE_model.encoder.load_weights(
                f"./checkpoints/Megabatch_checkpoint_{TYPE}_encoder_{self.checkpointname}"
            )
            self.AE_model.encoder.load_weights(
                f"./checkpoints/Megabatch_checkpoint_{TYPE}_decoder_{self.checkpointname}"
            )
        else:
            self.AE_model.load_weights(
                f"./checkpoints/Megabatch_checkpoint_{TYPE}{self.checkpointname}"
            )
            
        cats = []
        weights = []
        recon_err = []
        etmiss = []
        x = []
        
        test_cats = []
        test_weights = []
        test_recon_err = []
        test_etmiss = []
        test_x = []

        isBSM = []
        
        # * Validation inference for all megasets
        for megaset in range(self.totmegasets):
            print(f"Running inference on megabatch: {megaset}")

            FETCH_PATH = DATA_PATH / f"Megabatches/MB{megaset}"
            MERGE_PATH = FETCH_PATH / f"MergedMB{megaset}"

            #* Data 15 and data 16
            
            x_ = np.load(MERGE_PATH / f"Merged{megaset}_data15_and_16.npy")
            x_weights = np.load(MERGE_PATH / f"Merged{megaset}_data15_and_16_weights.npy")
            x_cats = np.load(MERGE_PATH / f"Merged{megaset}_data15_and_16_categories.npy")
            x_etmiss = np.load(MERGE_PATH / f"Merged{megaset}_data15_and_16_etmiss.npy")
            
            """x_ = np.load(MERGE_PATH / f"Merged{megaset}_xval.npy")
            x_weights = np.load(MERGE_PATH / f"Merged{megaset}_weights_val.npy")
            x_cats = np.load(MERGE_PATH / f"Merged{megaset}_categories_val.npy")
            x_etmiss = np.load(MERGE_PATH / f"Merged{megaset}_etmiss_val.npy")
            """
            x.append(x_)
            etmiss.append(x_etmiss)
            if run_inference:
                
                recon_err_back = self._inference(x_)
                recon_err.append(recon_err_back)    
            cats.append(x_cats)
            weights.append(x_weights)
            
            
            #* Data1516 mix
            test_x_ = np.load(MERGE_PATH / f"Merged{megaset}_data1516.npy")
            test_x_weights = np.load(MERGE_PATH / f"Merged{megaset}_data1516_weights.npy")
            test_x_cats = np.load(MERGE_PATH / f"Merged{megaset}_data1516_categories.npy")
            test_x_etmiss = np.load(MERGE_PATH / f"Merged{megaset}_data1516_etmiss.npy")

            test_x.append(test_x_)
            test_etmiss.append(test_x_etmiss)
            test_cats.append(test_x_cats)
            test_weights.append(test_x_weights)
            
            if run_inference:
            
                test_recon_err_back = self._inference(test_x_, types="signal")
                test_recon_err.append(test_recon_err_back) 
            
            isBSM_test = np.load(MERGE_PATH / f"Merged{megaset}_data1516_isBSM.npy")
            
            isBSM.append(isBSM_test)
            
            
           

        
        
        #* Data 15 and data 16
        etmiss = np.concatenate(etmiss, axis=0)
        x = np.concatenate(x, axis=0)
        isBSM = np.concatenate(isBSM, axis=0)
        
        signal = np.where(isBSM == 1)
        data = np.where(isBSM == 0)
        
        if run_inference:
            recon_err = np.concatenate(recon_err, axis=0)
            data_recon = test_recon_err[data]
        weights = np.concatenate(weights, axis=0)
        
        
        
        cats = np.concatenate(cats, axis=0)
        
        
        
        #* Data 1516 mix
        test_etmiss = np.concatenate(test_etmiss, axis=0)
        test_x = np.concatenate(test_x, axis=0)
        
        if run_inference: 
            test_recon_err = np.concatenate(test_recon_err, axis=0)
            
            
            signal_recon = test_recon_err[signal]
            
        test_weights = np.concatenate(test_weights, axis=0)
        test_cats = np.concatenate(test_cats, axis=0)
   
        
        
            
            
        test_x_sig = test_x[signal]
        
        
        data_weights = test_weights[data]
        data_cats = test_cats[data]
        
        signal_weights = test_weights[signal]
        signal_cats = test_cats[signal]
        
        
        if run_inference:
            signame = "Blind test"
            data_channel = ["Data 15 and 16"]
            #* Recon error plot
            plothisto = PlotHistogram(
                    STORE_IMG_PATH,
                    recon_err,
                    weights,
                    cats,
                    histoname="Reconstruction histogram with Data 15 and 16",
                    signal=test_recon_err,
                    signal_weights=test_weights,
                    signal_cats=test_cats,
                )
            plothisto.histogram_data(data_channel, sig_name=signame)

            #* Etmiss pre cut
            
            histo_tit = r"$e_T^{miss}$ distribution for Data 15 and 16 and " + f"{signame}"
            plotetmiss = PlotHistogram(
                STORE_IMG_PATH,
                etmiss,
                weights,
                cats,
                histoname=histo_tit,
                featurename=r"$e_T^{miss}$",
                histotitle=histo_tit,
                signal=test_etmiss,
                signal_weights=test_weights,
                signal_cats=test_cats,
            )
            plotetmiss.histogram_data(data_channel, sig_name=signame, etmiss_flag=True)
            
            
            #* Significance 
            
            small_sig = _significance_small(np.sum(test_weights), np.sum(weights))
            big_sig = _significance_big(np.sum(test_weights), np.sum(weights))

            print(" ")
            print(
                f"Pre cut etmiss;  Signifance small: {small_sig} | Significance big: {big_sig}"
            )
            print(" ")
            
            
            median = -2#np.median(plothisto.n_bins)
            std = np.abs(median / 5)
            print(f"Median recon: {median}, increment recon: {std}")


            for std_scale in range(1, 4):

                recon_er_cut = median + std_scale * std
                print(f"Recon err cut: {recon_er_cut}")

                error_cut_val = np.where(recon_err > (recon_er_cut))[0]
                error_cut_sig = np.where(test_recon_err > (recon_er_cut))[0]

                print(f"val cut shape: {np.shape(error_cut_val)}")

                val_weights_cut = weights[error_cut_val]
                sig_weights_cut = test_weights[error_cut_sig]

                val_cats_cut = cats[error_cut_val]
                signal_cats_cut = test_cats[error_cut_sig]

                etmiss_bkg = etmiss[error_cut_val]
                etmiss_sig = test_etmiss[error_cut_sig]

                small_sig = _significance_small(
                    np.sum(sig_weights_cut), np.sum(val_weights_cut)
                )
                big_sig = _significance_big(
                    np.sum(sig_weights_cut), np.sum(val_weights_cut)
                )

                print(" ")
                print(
                    f"S: {np.sum(sig_weights_cut):.3f} | B: {np.sum(val_weights_cut):.3f}"
                )
                print(f"Signifance small: {small_sig} | Significance big: {big_sig}")
                print(" ")

                

                etmiss_histoname = (
                    r"$e_T^{miss}$ with recon err cut of " + f"{recon_er_cut:.2f}"
                )

                plotetmiss_cut = PlotHistogram(
                    STORE_IMG_PATH,
                    etmiss_bkg,
                    val_weights_cut,
                    val_cats_cut,
                    histoname=etmiss_histoname,
                    featurename=r"$e_T^{miss}$",
                    histotitle=f"recon_errcut_{recon_er_cut:.2f}",
                    signal=etmiss_sig,
                    signal_weights=sig_weights_cut,
                    signal_cats=signal_cats_cut,
                )
                plotetmiss_cut.histogram_data(
                    data_channel, sig_name=signame, etmiss_flag=True
                )

                if not isinstance(plotetmiss_cut.n_bins, int):

                    small_sign, big_sign = bin_integrate_significance(
                        plotetmiss_cut.n_bins,
                        etmiss_bkg,
                        etmiss_sig,
                        val_weights_cut,
                        sig_weights_cut,
                    )

                    plt.plot(
                        plotetmiss_cut.n_bins,
                        small_sign,
                        "r-",
                        label=r"$\sqrt{2((s+b)log(1+\frac{s}{b}) - s)}$",
                    )
                    plt.plot(
                        plotetmiss_cut.n_bins,
                        big_sign,
                        "b-",
                        label=r"$\frac{s}{\sqrt{b}}$",
                    )
                    plt.legend()
                    plt.xlabel(r"$e_T^{miss}$ [GeV]", fontsize=25)
                    plt.ylabel("Signifiance", fontsize=25)
                    plt.legend(prop={"size": 15})
                    plt.title(r"Significance as function of $e_T^{miss}$", fontsize=25)
                    plt.savefig(
                        STORE_IMG_PATH
                        + f"histo/data/{TYPE}/{arc}/{SCALER}/significance_etmiss_{signame}_{recon_er_cut}.pdf"
                    )
                    plt.close() 
            
            
            print(np.sum(weights), np.shape(weights))
            print(np.sum(test_weights), np.shape(test_weights))
            
            #* Unblinded test
            
            
            
            signame="Unblind"
            
            plothisto_un = PlotHistogram(
                    STORE_IMG_PATH,
                    data_recon,
                    data_weights,
                    data_cats,
                    histoname="Reconstruction histogram unblinded",
                    signal=signal_recon,
                    signal_weights=signal_weights,
                    signal_cats=signal_cats,
                )
            plothisto_un.histogram_data(data_channel, sig_name=signame)
            
        
            
            print(f"Median recon: {median}, std recon: {std}")
            
        if create_rmm:
            plRMM = plotRMM(STORE_IMG_PATH, rmm_structure, RMMSIZE)
            plRMM.plotDfRmmMatrix(x, "data_15_and_16")
            plRMM.plotDfRmmMatrix(test_x, "data1516_mix")
            plRMM.plotDfRmmMatrix(test_x_sig, "signal")
            
            
    def _roc_curve(
        self,
        distribution_bkg,
        weights_bkg,
        distribution_sig,
        weights_sig,
        sig_name,
        figname,
    ):
        bkg_dist = distribution_bkg.copy()
        bkg = np.zeros(len(distribution_bkg))

        sig_dist = distribution_sig.copy()
        sg = np.ones(len(distribution_sig))

        label = np.concatenate((bkg, sg))
        scores = np.concatenate((bkg_dist, sig_dist))

        scaleFactor = np.sum(weights_sig) / np.sum(weights_bkg)

        weights = np.concatenate((weights_bkg * scaleFactor, weights_sig))

        fpr, tpr, thresholds = roc_curve(
            label, scores, sample_weight=weights, pos_label=1
        )
        sorted_index = np.argsort(fpr)
        fpr = np.array(fpr)[sorted_index]
        tpr = np.array(tpr)[sorted_index]

        roc_auc = auc(fpr, tpr)

        # RocCurveDisplay.from_predictions(label, scores, sample_weight=weights)
        plt.plot(fpr, tpr, label=f"AUC score: {roc_auc:.2f}")
        plt.xlabel("False positive rate", fontsize=25)
        plt.ylabel("True positive rate", fontsize=25)
        plt.legend(prop={"size": 15})
        plt.title(
            rf"ROC curve of {figname} for SM bkg and " + f"SUSY{sig_name}", fontsize=25
        )
        plt.savefig(
            STORE_IMG_PATH
            + f"histo/{LEP}/{TYPE}/{arc}/{SCALER}/roc_curve_{figname}_{sig_name}.pdf"
        )
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
                z_m, z_var, z = self.AE_model.encoder.predict(
                    arr, batch_size=self.b_size
                )
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


def write_to_file(file, string_write):
    with open(file, "a") as f:
        f.write(string_write)


def bin_integrate_significance(
    bins, dist_bkg, dist_sig, dist_bkg_weights, dist_sig_weights
):

    sign_bin_based_small = []
    sign_bin_based_big = []

    for bin in bins:

        bin_cond = np.where(dist_bkg > bin)[0]
        bin_cond_sig = np.where(dist_sig > bin)[0]

        s = dist_bkg_weights[bin_cond]
        s2 = dist_sig_weights[bin_cond_sig]
        w_bins_integrated_sum_bkg = np.sum(s)
        w_bins_integrated_sum_sig = np.sum(s2)

        sig_small = _significance_small(
            w_bins_integrated_sum_sig, w_bins_integrated_sum_bkg
        )
        sig_big = _significance_big(
            w_bins_integrated_sum_sig, w_bins_integrated_sum_bkg
        )

        sign_bin_based_big.append(sig_big)
        sign_bin_based_small.append(sig_small)

    return sign_bin_based_small, sign_bin_based_big


def _significance_small(s, b):
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))


def _significance_big(s, b):
    return s / np.sqrt(b)


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    global data_shape
    data_shape = 529

    L2 = LEP2ScaleAndPrep(DATA_PATH, True, SAVE_VAR, LOAD_VAR, lep=2, convert=True)

    #L2.RunTraining()

    #L2.RunInference()
    
    L2.RunBlindTest(False, True)
    
    #L2.checkIsBSM()
