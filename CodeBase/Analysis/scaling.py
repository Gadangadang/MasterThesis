import numpy as np
import pandas as pd
from os import listdir

import seaborn as sns
import tensorflow as tf
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from os.path import isfile, join


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


seed = tf.random.set_seed(1)


class ScaleAndPrep:
    def __init__(self, path: str) -> None:
        """_summary_

        Args:
            path (str): _description_
        """
        self.path = path
        self.onlyfiles = self.getDfNames()

        # self.scaleAndSplit()

    def getDfNames(self) -> Tuple[str, ...]:
        """
        Fetches all objects in a directory

        Returns:
            Tuple[str, ...]: list of pathnames
        """
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]

    def mergeDfs(self, exlude=["data18"]) -> None:
        """_summary_

        Args:
            exlude (list, optional): _description_. Defaults to ["data18"].
        """
        files = self.onlyfiles.copy()

        dfs = []

        categories = []

        for file in files:

            exl = [file.find(exl) for exl in exlude]

            df = pd.read_hdf(self.path / file)

            name = file[: file.find("_3lep")]

            count = len(df)
            names = [name] * count
            names = np.asarray(names)

            df.drop(
                [
                    "ele3_pt",
                    "ele3_eta",
                    "ele3_phi",
                    "ele3_m",
                    "muo3_pt",
                    "muo3_eta",
                    "muo3_phi",
                    "muo3_m",
                ],
                axis=1,
                inplace=True,
            )

            if sum(exl) > -1:
                self.data = df
                self.data["Category"] = names
                continue

            df["Category"] = names

            dfs.append(df)

        self.df = pd.concat(dfs)

    def scaleAndSplit(self):
        """_summary_
        """
        try:
            self.df
        except:
            self.mergeDfs()

        X_b_train, X_b_val = train_test_split(self.df, test_size=0.2, random_state=seed)

        self.weights_train = X_b_train["wgt_SG"]
        self.weights_val = X_b_val["wgt_SG"]
        self.data_weights = self.data["wgt_SG"]
        self.train_categories = X_b_train["Category"]
        self.val_categories = X_b_val["Category"]
        self.data_categories = self.data["Category"]

        print(self.train_categories.unique())

        X_b_train.drop("Category", axis=1, inplace=True)
        X_b_val.drop("Category", axis=1, inplace=True)
        self.data.drop("Category", axis=1, inplace=True)

        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

        with strategy.scope():

            scaler_ae = MinMaxScaler()  # StandardScaler()#MinMaxScaler()
            self.X_b_train = scaler_ae.fit_transform(X_b_train)
            self.X_b_val = scaler_ae.transform(X_b_val)

            self.data = scaler_ae.transform(self.data)


class RunAE:
    def __init__(self, data_structure):
        """
        Class to run training, inference and plotting

        Args:
            data_structure (object): Object containing the training, validation and test set
        """
        self.data_structure = data_structure
        self.X_train = self.data_structure.X_b_train
        self.X_val = self.data_structure.X_b_val
        self.data = self.data_structure.data
        self.data_shape = np.shape(self.X_train)[1]

    def getModel(self):
        """_summary_

        Returns:
            tf.python.keras.engine.functional.Functional: Model to use
        """
        inputs = tf.keras.layers.Input(shape=self.data_shape, name="encoder_input")
        x = tf.keras.layers.Dense(
            units=70,
            activation="tanh",
            kernel_regularizer=tf.keras.regularizers.L1(0.05),
            activity_regularizer=tf.keras.regularizers.L2(0.5),
        )(inputs)
        x_ = tf.keras.layers.Dense(units=45, activation="linear")(inputs)
        x1 = tf.keras.layers.Dense(
            units=20,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.L1(0.05),
            activity_regularizer=tf.keras.regularizers.L2(0.5),
        )(x_)
        val = 7
        x2 = tf.keras.layers.Dense(
            units=val, activation=tf.keras.layers.LeakyReLU(alpha=1)
        )(x1)
        encoder = tf.keras.Model(inputs, x2, name="encoder")

        latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")
        x = tf.keras.layers.Dense(
            units=22,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1(0.05),
            activity_regularizer=tf.keras.regularizers.L2(0.5),
        )(latent_input)
        x_ = tf.keras.layers.Dense(
            units=50, activation=tf.keras.layers.LeakyReLU(alpha=1)
        )(x)
        x1 = tf.keras.layers.Dense(
            units=73,
            activation="tanh",
            kernel_regularizer=tf.keras.regularizers.L1(0.05),
            activity_regularizer=tf.keras.regularizers.L2(0.5),
        )(x_)
        output = tf.keras.layers.Dense(self.data_shape, activation="linear")(x1)
        decoder = tf.keras.Model(latent_input, output, name="decoder")

        outputs = decoder(encoder(inputs))
        AE_model = tf.keras.Model(inputs, outputs, name="AE_model")

        hp_learning_rate = 0.0015
        optimizer = tf.keras.optimizers.Adam(hp_learning_rate)
        AE_model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

        # tf.keras.utils.plot_model(AE_model, to_file=path+"ae_model_plot.pdf", show_shapes=True, show_layer_names=True, expand_nested=True)

        return AE_model

    def trainModel(self):
        """_summary_
        """

        try:
            self.AE_model
        except:
            self.AE_model = self.getModel()

        print(type(self.AE_model))
        with tf.device("/GPU:0"):
            tf.config.optimizer.set_jit("autoclustering")
            self.AE_model.fit(
                self.X_train,
                self.X_train,
                epochs=10,
                batch_size=8192,
                validation_split=0.2,
                sample_weight=self.data_structure.weights_train,
            )

    def runInference(self):
        """
        """
        try:
            self.AE_model
        except:
            self.AE_model = self.getModel()

        with tf.device("/GPU:0"):
            self.pred_back = self.AE_model.predict(self.X_train, batch_size=8192)
            print("Background done")
            """
            pred_sig = self.AE_model.predict(test_set, batch_size=8192)
            print("Signal done")
            """

            self.pred_data = self.AE_model.predict(self.data, batch_size=8192)
            print("ATLAS data done")

        self.recon_err_back = self.reconstructionError(self.pred_back, self.X_train)
        self.recon_data = self.reconstructionError(self.pred_data, self.data)

    def reconstructionError(self, pred, real):
        """_summary_

        Args:
            pred (_type_): _description_
            real (_type_): _description_

        Returns:
            _type_: _description_
        """
        diff = pred - real
        err = np.power(diff, 2)
        err = np.sum(err, 1)
        err = np.log10(err)
        return err

    def checkReconError(self):
        """_summary_
        """

        # self.recon_err_back = self.recon_err_back.to_numpy()
        self.data_structure.weights_train = self.data_structure.weights_train.to_numpy()

        Zjets2 = self.recon_err_back[
            np.where(self.data_structure.train_categories == "Zjets2")
        ]
        diboson2L = self.recon_err_back[
            np.where(self.data_structure.train_categories == "diboson2L")
        ]
        diboson3L = self.recon_err_back[
            np.where(self.data_structure.train_categories == "diboson3L")
        ]
        diboson4L = self.recon_err_back[
            np.where(self.data_structure.train_categories == "diboson4L")
        ]
        triboson = self.recon_err_back[
            np.where(self.data_structure.train_categories == "triboson")
        ]
        higgs = self.recon_err_back[
            np.where(self.data_structure.train_categories == "higgs")
        ]
        singletop = self.recon_err_back[
            np.where(self.data_structure.train_categories == "singleTop")
        ]
        topOther = self.recon_err_back[
            np.where(self.data_structure.train_categories == "topOther")
        ]
        Wjets = self.recon_err_back[
            np.where(self.data_structure.train_categories == "Wjets")
        ]
        ttbar = self.recon_err_back[
            np.where(self.data_structure.train_categories == "ttbar")
        ]

        Zjets2_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "Zjets2")
        ]
        diboson2L_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "diboson2L")
        ]
        diboson3L_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "diboson3L")
        ]
        diboson4L_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "diboson4L")
        ]
        triboson_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "triboson")
        ]
        higgs_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "higgs")
        ]
        singletop_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "singleTop")
        ]
        topOther_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "topOther")
        ]
        Wjets_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "Wjets")
        ]
        ttbar_w = self.data_structure.weights_train[
            np.where(self.data_structure.train_categories == "ttbar")
        ]

        histo_atlas = [
            Zjets2,
            diboson2L,
            diboson3L,
            diboson4L,
            higgs,
            singletop,
            topOther,
            Wjets,
            ttbar,
            triboson,
        ]
        weight_atlas_data = [
            Zjets2_w,
            diboson2L_w,
            diboson3L_w,
            diboson4L_w,
            higgs_w,
            singletop_w,
            topOther_w,
            Wjets_w,
            ttbar_w,
            triboson_w,
        ]

        sum_w = [np.sum(weight) for weight in weight_atlas_data]
        sort_w = np.argsort(sum_w, kind="mergesort")

        print(sort_w)

        N, bins = np.histogram(
            self.recon_data, bins=25, weights=self.data_structure.data_weights
        )
        x = (np.array(bins[0:-1]) + np.array(bins[1:])) / 2

        plt.rcParams["figure.figsize"] = (12, 9)

        fig, ax = plt.subplots()

        n_bins = bins
        colors = [
            "green",
            "magenta",
            "blue",
            "red",
            "orange",
            "brown",
            "cyan",
            "mediumorchid",
            "chartreuse",
            "gold",
        ]
        labels = [
            "Zjets2",
            "diboson2L",
            "diboson3L",
            "diboson4L",
            "higgs",
            "singletop",
            "topOther",
            "Wjets",
            "ttbar",
            "triboson",
        ]

        print(np.asarray(labels, dtype=object)[sort_w])

        sns.set_style("darkgrid")
        ax.hist(
            np.asarray(histo_atlas, dtype=object)[sort_w],
            n_bins,
            density=False,
            stacked=True,
            alpha=0.5,
            histtype="bar",
            color=np.asarray(colors, dtype=object)[sort_w],
            label=np.asarray(labels, dtype=object)[sort_w],
            weights=np.asarray(weight_atlas_data, dtype=object)[sort_w],
        )

        ax.scatter(x, N, marker="+", label="Data", color="black")

        ax.legend(prop={"size": 20})
        ax.set_title(
            "Reconstruction error histogram with background and ATLAS data", fontsize=25
        )
        ax.set_xlabel("Log10 Reconstruction Error", fontsize=25)
        ax.set_ylabel("#Events", fontsize=25)
        # ax.set_xlim([0, 3.5])
        ax.set_ylim([1, 5e6])
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=25)
        fig.tight_layout()
        plt.savefig(path + "/b_data_recon_big_rm3_feats.pdf")
        plt.close()


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    path = "/home/sgfrette/MasterThesis/Figures/testing/"
    pat = Path("/storage/William_Sakarias/Sakarias_Data")

    sp = ScaleAndPrep(pat)
    sp.scaleAndSplit()

    rae = RunAE(sp)
    rae.trainModel()
    rae.runInference()
    rae.checkReconError()
