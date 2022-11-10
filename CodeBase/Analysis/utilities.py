import time
import random
import requests
import numpy as np
import pandas as pd
from config import *
import seaborn as sns
from pathfile import *
from os import listdir
from numpy import array
import tensorflow as tf
import keras_tuner as kt
from pathlib import Path
from typing import Tuple
import plotly.express as px
import matplotlib.pyplot as plt
from os.path import isfile, join
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

seed = tf.random.set_seed(1)

tf.keras.utils.get_custom_objects()["leaky_relu"] = tf.keras.layers.LeakyReLU()


scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}


class plotRMM:
    def __init__(self, path: Path, rmm_structure: dict, N_row: int):
        self.path = path
        self.rmm_structure = rmm_structure
        self.N_row = N_row
        self.onlyfiles = self.getDfNames()

    def getDfNames(self) -> Tuple[str, ...]:
        """
        Fetches all objects in a directory

        Returns:
            Tuple[str, ...]: list of pathnames
        """
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]  # type: ignore

    def plotRMM(self):
        """_summary_"""

        print("*** Plotting starting ***")

        for idx, file in enumerate(self.onlyfiles):
            file_idx = file.find("_3lep")
            df = pd.read_hdf(self.path / file)
            print(file[:file_idx])
            self.plotDfRmmMatrix(df, file[:file_idx])

        print("*** Plotting done ***")

    def plotDfRmmMatrix(self, df: pd.DataFrame, process: str) -> None:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            process (str): _description_
        """

        col = len(df.columns)
        row = len(df)

        df2 = df.mean()

        tot = len(df2)
        row = int(np.sqrt(tot))

        rmm_mat = np.zeros((row, row))

        df2 = df2.to_numpy()

        p = 0

        for i in range(row):
            for j in range(row):
                rmm_mat[i, j] = df2[p]
                p += 1

        names = [" "]

        for i in range(1, self.N_row):
            name = self.rmm_structure[i][0]
            names.append(name)

        # rmm_mat[rmm_mat < 0.00009] = np.nan

        rmm_mat[rmm_mat == 0] = np.nan

        fig = px.imshow(
            rmm_mat,
            labels=dict(x="Particles", y="Particles", color="Intensity"),
            x=names,
            y=names,
            aspect="auto",
            color_continuous_scale="Viridis",
            text_auto=".3f",
        )
        fig.update_xaxes(side="top")

        fig.write_image(f"../../Figures/testing/rmm/rmm_avg_{process}.pdf")

    def plotDfRmmMatrixNoMean(self, df: pd.DataFrame, process: str, idx: int) -> None:
        """

        Args:
            df (pd.DataFrame): _description_
            process (str): _description_
            idx (int): _description_
        """

        try:  # In case pandas dataframe is passed
            col = len(df.columns)
            df2 = df.iloc[idx].to_numpy()
        except:  # In case regular numpy array is passed
            col = len(df)
            df2 = df[idx]

        row = len(df)

        tot = len(df2)
        row = int(np.sqrt(tot))

        rmm_mat = np.zeros((row, row))

        p = 0

        for i in range(row):
            for j in range(row):
                rmm_mat[i, j] = df2[p]
                p += 1

        names = [" "]

        for i in range(1, self.N_row):
            name = self.rmm_structure[i][0]
            number = name[-1]
            part_type = name[:-2]
            name = rf"${part_type}_{number}$"

            names.append(name)

        # rmm_mat[rmm_mat < 0.00009] = np.nan

        rmm_mat[rmm_mat == 0] = np.nan

        fig = px.imshow(
            rmm_mat,
            labels=dict(x="Particles", y="Particles", color="Intensity"),
            x=names,
            y=names,
            aspect="auto",
            color_continuous_scale="Viridis",
            text_auto=".3f",
            title=f"Event {idx} channel {process}",
        )
        fig.update_xaxes(side="top")

        fig.write_image(f"../../Figures/testing/rmm/rmm_event_{idx}_{process}.pdf")


class ScaleAndPrep:
    def __init__(self, path: Path, event_rmm=False, save=False, load=False) -> None:
        """_summary_

        Args:
            path (str): _description_
            event_rmm (bool, optional):
        """
        self.path = path
        self.onlyfiles = self.getDfNames()
        self.event_rmm = event_rmm
        self.load = load
        self.save = save
        # self.scaleAndSplit()

    def getDfNames(self) -> Tuple[str, ...]:
        """
        Fetches all objects in a directory

        Returns:
            Tuple[str, ...]: list of pathnames
        """

        files = [
            f
            for f in listdir(self.path)
            if isfile(join(self.path, f))
            and f[-4:] != ".npy"
            and f[-4:] != ".csv"
            and f[-5:] != "_b.h5"
            and f[-4:] != ".txt"
        ]

        return files  # type: ignore

    def fetchDfs(self) -> None:  # exlude=["data18", "ttbar"]
        """
        This function takes all dataframes stored as hdf5 files and adds them to a list,
        where this list later is used for scaling, splitting and merging of dataframes.

        """
        files = self.onlyfiles.copy()  # type: ignore

        self.dfs = []
        self.datas = []
        self.signals = []

        data_names = ["data15", "data16", "data17", "data18"]
        signal_names = [
            "LRSMWR2400NR50",
            "LRSMWR4500NR400",
            "WeHNL5040Glt01ddlepfiltch1",
            "WeHNL5060Glt01ddlepfiltch1",
            "WeHNL5070Glt01ddlepfiltch1",
            "WmuHNL5040Glt01ddlepfiltch1",
            "WmuHNL5060Glt01ddlepfiltch1",
            "WmuHNL5070Glt01ddlepfiltch1",
        ]

        charge_df_data = []

        self.tot_mc_events = 0
        self.tot_data_events = 0
        self.tot_signal_events = 0

        for file in files:

            df = pd.read_hdf(self.path / file)

            name = file[: file.find("_3lep")]

            if name in data_names:
                name = "data"

            try:

                df.drop(
                    [
                        "nlep_BL",
                        "nlep_SG",
                        "ele_0_charge",
                        "ele_1_charge",
                        "ele_2_charge",
                        "muo_0_charge",
                        "muo_1_charge",
                        "muo_2_charge",
                    ],
                    axis=1,
                    inplace=True,
                )
            except:
                pass

            count = len(df)
            names = [name] * count
            names = np.asarray(names)

            df["Category"] = names

            if name == "data":
                self.datas.append(df)
                self.tot_data_events += len(df)
            elif name in signal_names:
                self.signals.append(df)
                self.tot_signal_events += len(df)
            else:
                self.dfs.append(df)
                self.tot_mc_events += len(df)

    def MergeScaleAndSplit(self):
        """_summary_"""

        if not self.load:
            plotRMMMatrix = plotRMM(self.path, rmm_structure, 9)

            try:
                self.df  # type: ignore
            except:
                self.fetchDfs()

            df_train = []
            df_val = []

            df_train_w = []
            df_val_w = []

            df_train_cat = []
            df_val_cat = []

            for df in self.dfs:
                weight = df["wgt_SG"]
                print(df["Category"].unique())

                print(np.sum(weight))
                flag = 1
                while flag:
                    x_b_train, x_b_val = train_test_split(
                        df, test_size=0.2, random_state=seed
                    )

                    weights_train = x_b_train["wgt_SG"]
                    weights_val = x_b_val["wgt_SG"]

                    ratio = np.sum(weights_train) / np.sum(weight) * 100

                    if ratio < 81 and ratio > 79:
                        print(np.sum(weights_train))
                        print(f"Ratio: {ratio:.2f}%")
                        break

                train_categories = x_b_train["Category"]
                val_categories = x_b_val["Category"]

                df_train.append(x_b_train)
                df_train_w.append(weights_train)
                df_val.append(x_b_val)
                df_val_w.append(weights_val)

                df_train_cat.append(train_categories)
                df_val_cat.append(val_categories)

            X_b_train = pd.concat(df_train)
            X_b_val = pd.concat(df_val)

            self.train_categories = pd.concat(df_train_cat)
            self.val_categories = pd.concat(df_val_cat)

            self.weights_train = pd.concat(df_train_w)
            self.weights_val = pd.concat(df_val_w)

            self.data = pd.concat(self.datas)

            self.data_categories = self.data["Category"]

            self.data_weights = self.data["wgt_SG"]

            channels = [
                "Zeejets",
                "Zmmjets",
                "Zttjets",
                "diboson2L",
                "diboson3L",
                "diboson4L",
                "higgs",
                "singletop",
                "topOther",
                "Wjets",
                "triboson",
                "ttbar",
            ]

            # Indentifying Zeejets events for rmm single event plotting
            idx_rmm = []
            choices = random.sample(channels, 4)
            for choice in choices:

                id_rmm = np.where(self.train_categories == choice)[0]

                idx_rmm.append((choice, id_rmm))

            self.idxs = []

            for channel in channels:

                idx_val = np.where(X_b_val["Category"] == channel)[0]
                idx_train = np.where(X_b_train["Category"] == channel)[0]
                id = (channel, idx_train, idx_val)

                self.idxs.append(id)

            X_b_train.drop("Category", axis=1, inplace=True)
            X_b_val.drop("Category", axis=1, inplace=True)
            self.data.drop("Category", axis=1, inplace=True)

            X_b_train.drop("wgt_SG", axis=1, inplace=True)
            X_b_val.drop("wgt_SG", axis=1, inplace=True)
            self.data.drop("wgt_SG", axis=1, inplace=True)

            # print(X_b_train.columns)

            cols = [
                "e_T_miss",
                "m_T_jet_0",
                "m_T_jet_1",
                "m_T_ele_0",
                "m_T_ele_1",
                "m_T_ele_2",
                "m_T_muo_0",
                "m_T_muo_1",
                "m_T_muo_2",
                "e_T_jet_0",
                "m_jet_0_jet_1",
                "m_jet_0_ele_0",
                "m_jet_0_ele_1",
                "m_jet_0_ele_2",
                "m_jet_0_muo_0",
                "m_jet_0_muo_1",
                "m_jet_0_muo_2",
                "m_jet_1_ele_0",
                "m_jet_1_ele_1",
                "m_jet_1_ele_2",
                "m_jet_1_muo_0",
                "m_jet_1_muo_1",
                "m_jet_1_muo_2",
                "e_T_ele_0",
                "m_ele_0_ele_1",
                "m_ele_0_ele_2",
                "m_ele_0_muo_0",
                "m_ele_0_muo_1",
                "m_ele_0_muo_2",
                "m_ele_1_ele_2",
                "m_ele_1_muo_0",
                "m_ele_1_muo_1",
                "m_ele_1_muo_2",
                "m_ele_2_muo_0",
                "m_ele_2_muo_1",
                "m_ele_2_muo_2",
                "e_T_muo_0",
                "m_muo_0_muo_1",
                "m_muo_0_muo_2",
                "m_muo_1_muo_2",
                "flcomp",
            ]

            scaler = scalers[SCALER]
            """
            column_trans = ColumnTransformer(
                    [('scaler_ae', scaler, cols)],
                    remainder='passthrough'
                )
            """
            column_trans = scaler

            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            with strategy.scope():

                self.X_b_train = column_trans.fit_transform(X_b_train)
                self.X_b_val = column_trans.transform(X_b_val)
                self.data = column_trans.transform(self.data)

            if self.event_rmm:
                for choice, idxss in idx_rmm:
                    event = random.sample(list(idxss), 1)[0]
                    # print(event, choice)
                    plotRMMMatrix.plotDfRmmMatrixNoMean(self.X_b_train, choice, event)

            ### Plot RMM for each channel

            for channel, idx_train, idx_val in self.idxs:
                cols = X_b_train.columns
                train = self.X_b_train[idx_train]
                val = self.X_b_val[idx_val]
                train = pd.DataFrame(data=train, columns=cols)
                val = pd.DataFrame(data=val, columns=cols)
                plot_df = pd.concat([train, val])
                plotRMMMatrix.plotDfRmmMatrix(plot_df, channel)

            datapoint_string = f"""
    Total MC events: {self.tot_mc_events} \n
    Total Data events: {self.tot_data_events} \n
    Total Signal events: {self.tot_signal_events} \n
                """
            print(datapoint_string)

            if self.save:
                np.save(DATA_PATH / "X_train.npy", self.X_b_train)
                np.save(DATA_PATH / "X_val.npy", self.X_b_val)
                np.save(DATA_PATH / "Data.npy", self.data)

                np.save(DATA_PATH / "channel_names.npy", np.asarray(channels))
                for row in self.idxs:
                    name = row[0]
                    np.save(DATA_PATH / f"{name}_train_idxs.npy", row[1])
                    np.save(DATA_PATH / f"{name}_val_idxs.npy", row[2])

                # Using _b to separate these hdf5 files from the sample files
                self.train_categories.to_hdf(DATA_PATH / "train_cat_b.h5", "mini")
                self.val_categories.to_hdf(DATA_PATH / "val_cat_b.h5", "mini")

                self.weights_train.to_hdf(DATA_PATH / "train_weight_b.h5", "mini")
                self.weights_val.to_hdf(DATA_PATH / "val_weight_b.h5", "mini")

                self.data_categories.to_hdf(DATA_PATH / "data_cat_b.h5", "mini")
                self.data_weights.to_hdf(DATA_PATH / "data_weight_b.h5", "mini")

        else:
            self.X_b_train = np.load(DATA_PATH / "X_train.npy")
            self.X_b_val = np.load(DATA_PATH / "X_val.npy")
            self.data = np.load(DATA_PATH / "Data.npy")

            self.idxs = []
            channels = np.load(DATA_PATH / "channel_names.npy")
            for name in channels:

                train = np.load(DATA_PATH / f"{name}_train_idxs.npy")
                val = np.load(DATA_PATH / f"{name}_val_idxs.npy")
                self.idxs.append((name, train, val))

            # print(self.idxs)

            # Using _b to separate these hdf5 files from the sample files
            self.train_categories = pd.read_hdf(DATA_PATH / "train_cat_b.h5")
            self.val_categories = pd.read_hdf(DATA_PATH / "val_cat_b.h5")

            self.weights_train = pd.read_hdf(DATA_PATH / "train_weight_b.h5")
            self.weights_val = pd.read_hdf(DATA_PATH / "val_weight_b.h5")

            self.data_categories = pd.read_hdf(DATA_PATH / "data_cat_b.h5")
            self.data_weights = pd.read_hdf(DATA_PATH / "data_weight_b.h5")


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
        self.data = self.data_structure.data
        self.data_shape = np.shape(self.X_train)[1]
        self.idxs = self.data_structure.idxs
        self.val_cats = self.data_structure.val_categories.to_numpy()
        self.err_val = self.data_structure.weights_val.to_numpy()
        self.err_train = self.data_structure.weights_train.to_numpy()

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
        except:
            self.AE_model = self.getModel()

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
        except:
            if tuned_model:
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

        with tf.device("/GPU:0"):
            self.pred_back = self.AE_model.predict(X_val, batch_size=self.b_size)
            self.recon_err_back = self.reconstructionError(self.pred_back, X_val)
            print("Background done")

            if len(test_set) > 0:
                self.pred_sig = self.AE_model.predict(test_set, batch_size=self.b_size)
                self.recon_sig = self.reconstructionError(self.pred_sig, test_set)
                print("Signal done")

            self.pred_data = self.AE_model.predict(self.data, batch_size=self.b_size)
            self.recon_data = self.reconstructionError(self.pred_data, self.data)
            print("ATLAS data done")

    def reconstructionError(self, pred: np.ndarray, real: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            pred (np.ndarray): _description_
            real (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        diff = pred - real
        err = np.power(diff, 2)
        err = np.sum(err, 1)
        err = np.log10(err)
        return err

    def checkReconError(self, channels: list, sig_name="nosig")->None:
        """_summary_

        Args:
            channels (list): _description_
            sig_name (str, optional): _description_. Defaults to "nosig".
        """

        histo_atlas = []
        weight_atlas_data = []
        try:
    
            for id, channel in enumerate(channels):
                
                idxs = self.tot_weights_per_channel[id]
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
        except:
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
        ax.set_title(
            "Reconstruction error histogram with MC and ATLAS data", fontsize=25
        )
        ax.set_xlabel("Log10 Reconstruction Error", fontsize=25)
        ax.set_ylabel("#Events", fontsize=25)
        # ax.set_xlim([0, 3.5])
        ax.set_ylim([0.1, 5e6])  # type: ignore
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=25)
        fig.tight_layout()
        plt.savefig(self.path + f"histo/b_data_recon_big_rm3_feats_sig_{sig_name}.pdf")
        plt.close()

class HyperParameterTuning(RunAE):
    def __init__(self, data_structure: object, path: str)->None:
        super().__init__(data_structure, path)
        
    def runHpSearch(
        self, X_train: np.ndarray, X_val: np.ndarray, sample_weight: dict, small=False, epochs=20
    )->None:
        """_summary_"""

        device_lib.list_local_devices()
        tf.config.optimizer.set_jit("autoclustering")
        with tf.device("/GPU:0"):
            if small:
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
        self, X_b: np.ndarray, X_back_test: np.ndarray, sample_weight: dict
    ) -> None:
        """_summary_

        Args:
            X_b (np.ndarray): _description_
            X_back_test (np.ndarray): _description_
        """
        tuner = kt.Hyperband(
            self.AE_model_builder,
            objective=kt.Objective("val_mse", direction="min"),
            max_epochs=20,
            factor=3,
            directory="GridSearches",
            project_name="AE",
            overwrite=True,
        )
        print(tuner.search_space_summary())

        tuner.search(
            X_b,
            X_b,
            epochs=20,
            batch_size=self.b_size,
            validation_data=(X_back_test, X_back_test),
            sample_weight=sample_weight,
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print(tuner.search_space_summary())

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
            units=hp.Int("num_of_neurons1", min_value=10, max_value=29, step=1),
            activation=activations.get(
                hp.Choice("1_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(inputs)

        val = hp.Int("lat_num", min_value=1, max_value=9, step=1)

        # Forth hidden layer
        x2 = tf.keras.layers.Dense(
            units=val,
            activation=activations.get(
                hp.Choice("2_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x1)

        # Encoder definition
        encoder = tf.keras.Model(inputs, x2, name="encoder")

        # Latent space
        latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")

        # Output layer
        output = tf.keras.layers.Dense(
            self.data_shape,
            activation=activations.get(
                hp.Choice("3_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(latent_input)

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


class ChannelTraining(RunAE):
    def __init__(self,data_structure:object, path:str)->None:
        super().__init__(data_structure, path)
        
    def run(self, small=False)->None:
        """_summary_

        Args:
            small (bool, optional): _description_. Defaults to False.
        """
        st = time.time()
        
        self.data_structure.weights_val = self.data_structure.weights_val.to_numpy()

        for channel, idx_train, idx_val in self.idxs:

            channels = self.channels.copy()
            channels.remove(channel)

            print(f"Channel: {channel}  started")

            new_index = np.delete(np.asarray(range(len(self.X_train))), idx_train)
            new_index_val = np.delete(np.asarray(range(len(self.X_val))), idx_val)

            self.val_cats = self.data_structure.val_categories.to_numpy().copy()[
                new_index_val
            ]

            self.err_val = self.data_structure.weights_val.copy()[new_index_val]

            X_train_reduced = self.X_train.copy()[new_index]
            X_val_reduced = self.X_val.copy()[new_index_val]

            sample_weight = self.data_structure.weights_train.to_numpy().copy()[
                new_index
            ]
            sample_weight = pd.DataFrame(sample_weight)

            channel_train_set = self.X_train.copy()[idx_train]
            channel_val_set = self.X_val.copy()[idx_val]

            signal = channel_val_set  # np.concatenate((channel_train_set, channel_val_set), axis=0)

            sig_err_t = self.data_structure.weights_train.to_numpy()[
                np.where(self.data_structure.train_categories == channel)[0]
            ]

            sig_err_v = self.data_structure.weights_val[
                np.where(self.data_structure.val_categories == channel)[0]
            ]

            self.sig_err = sig_err_v  # np.concatenate((sig_err_t, sig_err_v), axis=0)

            self.name = "no_" + channel

            HPT = HyperParameterTuning(self.data_structure, STORE_IMG_PATH)
            HPT.runHpSearch(
                X_train_reduced, X_val_reduced, sample_weight, small=small, epochs=2
            )

            # self.trainModel(X_train_reduced, X_val_reduced, sample_weight)
            print(" ")
            print("Hyperparam search done")
            print(" ")

            self.trainModel(X_train_reduced, X_val_reduced, sample_weight)

            self.runInference(X_val_reduced, signal, True)

            self.checkReconError(channels, sig_name=channel)
            
            et = time.time()

            img_path = Path(f"histo/b_data_recon_big_rm3_feats_sig_{channel}.pdf")
            path = STORE_IMG_PATH/img_path

            files = {"photo":open(path, "rb")}
            message = f"Done calculating dummy data plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
            resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
            print(resp.status_code)

class OnePercentData(RunAE):
    def __init__(self, data_structure:object, path: str)->None:
        super().__init__(data_structure, path)
        
    def run(self)->None:
        st = time.time()
        
        #* Fetch events from the MC set
        
        self.tot_set_train_idxs_list = []
        self.tot_set_val_idxs_list = []
        
        for channel, idx_train, idx_val in self.idxs:
            print(f"Channel: {channel}")
            
            nr_channel_events_train = len(idx_train)
            one_percent_no_events_t = int(nr_channel_events_train/100)
            new_indices_train = np.random.choice(idx_train, size=one_percent_no_events_t, replace=False)  
            
            weights = self.err_train[idx_val].copy()
            
            one_percent_weights = self.err_train[new_indices_train].copy()
            
            #* Fetch randomly selected events for training
            flag = True
            while flag:
                nr_channel_events_train = len(idx_train)
                one_percent_no_events_t = one_percent_no_events_t + 15
                print(one_percent_no_events_t)
                new_indices_train = np.random.choice(idx_train, size=one_percent_no_events_t, replace=False)  
                
                weights = self.err_train[idx_train].copy()
                
                one_percent_weights = self.err_train[new_indices_train].copy()
                
                
                
                if np.abs(np.sum(one_percent_weights) - (np.sum(weights)/100)) > 100:
                    one_percent_no_events_t = int(nr_channel_events_train/100)
                if np.abs(np.sum(one_percent_weights) - (np.sum(weights)/100)) < 1:
                    flag = False
            
            self.tot_set_train_idxs_list.append(new_indices_train)
                        
            nr_channel_events_val= len(idx_val)
            one_percent_no_events_v = int(nr_channel_events_val/100)
            weights_val = self.err_val[idx_val].copy()
            
            new_indices_val = np.random.choice(idx_val, size=one_percent_no_events_v, replace=False) 
            
            one_percent_weights = self.err_train[new_indices_val].copy()
            
            #* Fetch randomly selected events for validation
            flag2 = True
            while flag2:
                nr_channel_events_val = len(idx_val)
                one_percent_no_events_v = one_percent_no_events_v + 1
                new_indices_val = np.random.choice(idx_val, size=one_percent_no_events_v, replace=False)  
                
                weights_val = self.err_val[idx_val].copy()
                
                one_percent_weights = self.err_val[new_indices_val].copy()
                
                
                if np.abs(np.sum(one_percent_weights) - (np.sum(weights_val)/100)) > 50:
                    one_percent_no_events_v = int(nr_channel_events_val/100)
                    
                if np.abs(np.sum(one_percent_weights) - (np.sum(weights_val)/100)) < 1:
                    flag2 = False
            
            self.tot_set_val_idxs_list.append(new_indices_val)
            
        print(f"Number of selected events: {one_percent_no_events_t + one_percent_no_events_v}")    
            
        self.tot_set_train_idxs = np.concatenate(self.tot_set_train_idxs_list, axis=0)
        self.tot_set_val_idxs = np.concatenate(self.tot_set_val_idxs_list, axis=0)
        
        self.tot_weights_per_channel = []
        self.tot_data = []
        
        self.act_weights = []
        start = 0
        
        for id, idx_train  in enumerate(self.tot_set_train_idxs_list):
            idx_val = self.tot_set_val_idxs_list[id]
            
            x_tot = np.concatenate((self.X_train[idx_train], self.X_val[idx_val]), axis = 0)
            self.tot_data.append(x_tot)
            
            act = np.concatenate((self.data_structure.weights_train.to_numpy().copy()[idx_train], 
             self.data_structure.weights_val.to_numpy().copy()[idx_val]), axis=0)
            
            self.act_weights.append(act)
            idxs = np.concatenate((idx_train, idx_val), axis=0)
            
            end = start + len(idxs)
            
            print(len(x_tot), len(idxs))
            
            self.tot_weights_per_channel.append(np.asarray(range(start, end)))
            
            start = end
            
        
        X_tot = np.concatenate(self.tot_data, axis=0)
        self.act_weights = np.concatenate(self.act_weights, axis=0)
        
        X_train = self.X_train[self.tot_set_train_idxs]
        X_val = self.X_val[self.tot_set_val_idxs]
        
        sample_weight_n = self.data_structure.weights_train.to_numpy().copy()[
            self.tot_set_train_idxs
        ]
        
        sample_weight = pd.DataFrame(sample_weight_n)

        self.err_val = self.act_weights
        
        
        #* Fetch events from the data
        nr_data_events = len(self.data) 
        indices = range(nr_data_events)
        one_percent_no_events = int(nr_data_events/100)
        new_indices = np.random.choice(indices, size=one_percent_no_events, replace=False)        
        
        self.dummysample_dataset = self.data[new_indices]
        self.sig_err = np.ones(len(self.dummysample_dataset))
        
        #* Check weight comparison
        print(" ")
        print("*****************************************")
        print(f"Data weights: {np.sum(self.sig_err):.1f} | MC weights: {np.sum(self.err_val):.1f}")
        print("*****************************************")
        print(" ")
        
        
        #* Tuning, training, and inference
        HPT = HyperParameterTuning(self.data_structure, STORE_IMG_PATH)
        HPT.runHpSearch(
            X_train, X_val, sample_weight, small=False
        )
        

        self.trainModel(X_train, X_val, sample_weight)

        
        self.runInference(X_tot, self.dummysample_dataset, True)

        self.checkReconError(self.channels, sig_name="1%_ATLAS_Data")
        
        et = time.time()
        
        img_path = Path("histo/b_data_recon_big_rm3_feats_sig_1%_ATLAS_Data.pdf")
        path = STORE_IMG_PATH/img_path

        files = {"photo":open(path, "rb")}
        message = f"Done calculating dummy data plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
        resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
        print(resp.status_code)
        
        
class DummyData(RunAE):
    def __init__(self, data_structure:object, path:str)->None:
        super().__init__(data_structure, path)
        
    def swapEventsInChannels(self, fraction_cols:float, fraction_swap:float)->None:
        st = time.time()
        
        rows, cols = np.shape(self.X_val)
        
        nr_rows_swap = int(rows*fraction_swap)
        
        nr_cols_swap = int(cols*fraction_cols)
        
        cols_to_swap = np.random.choice(range(cols), size=nr_cols_swap, replace=False)
        
        rows_to_swap = np.random.choice(range(rows), size=nr_rows_swap, replace=False)
        
        old_row = rows_to_swap.copy()
        
        np.random.shuffle(rows_to_swap)
        
        pairs = []
        
        train_cat = self.data_structure.train_categories.to_numpy()
        val_cat = self.data_structure.val_categories.to_numpy()
        
        for i in range(0, len(rows_to_swap), 2):
            
            col = np.random.choice(cols_to_swap, size=1, replace=False)[0]
            #print(col)
            pairs.append((col, rows_to_swap[i], rows_to_swap[i+1]))   
        
        X_val_dummy = self.X_val.copy()
        
        for column_number in cols_to_swap:
            
            np.random.shuffle(rows_to_swap)
        
            X_val_dummy[old_row, column_number] = X_val_dummy[rows_to_swap, column_number]
            
        val_cat[old_row] = "Signal" 
            
        

            #print(X_val_dummy[row_1, column_number], X_val_dummy[row_2, column_number])
        
        val_cat = np.concatenate((train_cat, val_cat), axis=0)
      
        
        X_tot = np.concatenate((self.X_train, X_val_dummy), axis=0)
        
      
        
        
        signal = X_tot[np.where(val_cat == "Signal")]
        X_tot = X_tot[np.where(val_cat != "Signal")]
        
        
        
        sample_weight_t = self.data_structure.weights_train.to_numpy().copy()
        sample_weight_v = self.data_structure.weights_val.to_numpy().copy()
        
        sample_weight = pd.DataFrame(sample_weight_t)
        
        self.err_val = np.concatenate((sample_weight_t, sample_weight_v), axis=0)
        
        self.sig_err = self.err_val[np.where(val_cat == "Signal")]
        self.err_val = self.err_val[np.where(val_cat != "Signal")]
        
        self.val_cats = np.concatenate((train_cat, val_cat), axis=0)
        self.val_cats = self.val_cats[np.where(val_cat != "Signal")]
        
         #* Tuning, training, and inference
        HPT = HyperParameterTuning(self.data_structure, STORE_IMG_PATH)
        HPT.runHpSearch(
            self.X_train, X_val_dummy, sample_weight, small=False, epochs=3
        )
        

        self.trainModel(self.X_train, X_val_dummy, sample_weight)

        
        self.runInference(X_tot, signal,True)

       
        self.checkReconError(self.channels, sig_name="Dummydata")   
        
        et = time.time()
        
        img_path = Path("histo/b_data_recon_big_rm3_feats_sig_Dummydata.pdf")
        path = STORE_IMG_PATH/img_path

        files = {"photo":open(path, "rb")}
        message = f"Done calculating dummy data plot, took {et-st:.1f}s or {(et-st)/60:.1f}m"
        resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={chat_id}&caption={message}", files=files)
        print(resp.status_code)
                    
            
            
            
          
            
      