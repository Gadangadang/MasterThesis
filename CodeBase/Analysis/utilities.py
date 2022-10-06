import matplotlib
import numpy as np
import pandas as pd
from os import listdir

import seaborn as sns
import tensorflow as tf
import keras_tuner as kt
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from os.path import isfile, join

from sklearn import preprocessing
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

seed = tf.random.set_seed(1)

rmm_structure = {
    1: [
        "jet_0",
        "jetPt[jet_SG]",
        "jetEta[jet_SG]",
        "jetPhi[jet_SG]",
        "jetM[jet_SG]",
        0,
    ],
    2: [
        "jet_1",
        "jetPt[jet_SG]",
        "jetEta[jet_SG]",
        "jetPhi[jet_SG]",
        "jetM[jet_SG]",
        1,
    ],
    3: [
        "ele_0",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        0,
    ],
    4: [
        "ele_1",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        1,
    ],
    5: [
        "ele_2",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        2,
    ],
    6: [
        "muo_0",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        0,
    ],
    7: [
        "muo_1",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        1,
    ],
    8: [
        "muo_2",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        2,
    ],
}




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
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]

    def plotRMM(self):

        print("*** Plotting starting ***")

        for idx, file in enumerate(self.onlyfiles):
            file_idx = file.find("_3lep")
            df = pd.read_hdf(self.path / file)
            print(file[:file_idx])
            self.plotDfRmmMatrix(df, file[:file_idx])

        print("*** Plotting done ***")

    def plotDfRmmMatrix(self, df: pd.DataFrame, process: str) -> None:

        col = len(df.columns)
        row = len(df)

        print("")
        print(f"Size: {row}")
        print("")
        df2 = df.mean()

        tot = len(df2)
        row = int(np.sqrt(tot))
        print(row)

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

        fig, ax = plt.subplots()

        im, cbar = self.heatmap(rmm_mat, names, names, ax=ax, cbarlabel="Intensity")
        texts = self.annotateHeatmap(im, valfmt="{x:.3f}")

        im = ax.imshow(rmm_mat)

        fig.tight_layout()

        plt.savefig(f"../../Figures/testing/rmm_avg_{process}.pdf")
        plt.close()

    def heatmap(
        self, data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs
    ):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # Turn spines off and create white grid.
        # ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotateHeatmap(
        self,
        im,
        data=None,
        valfmt="{x:.2f}",
        textcolors=("black", "white"),
        threshold=None,
        **textkw,
    ):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.0

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center", verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

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

    def fetchDfs(self, exlude=["ttbar"]) -> None: #exlude=["data18", "ttbar"]
        """
        This function takes all dataframes stored as hdf5 files and adds them to a list,
        where this list later is used for scaling, splitting and merging of dataframes.


        Args:
            exlude (list, optional): _description_. Defaults to ["data18"].
        """
        files = self.onlyfiles.copy()

        self.dfs = []

        

        for file in files:
            

            exl = [file.find(exl) for exl in exlude]

            df = pd.read_hdf(self.path / file)

            name = file[: file.find("_3lep")]
            
            
            
            if name == "data18":
                continue

            count = len(df)
            names = [name] * count
            names = np.asarray(names)
            
            if name == "singletop":
                print(np.shape(df))

            try:
                df.drop(
                    [
                    'ele_0_charge', 
                    'ele_1_charge', 
                    'ele_2_charge', 
                    'muo_0_charge',
                    'muo_1_charge', 
                    'muo_2_charge'
                    ],
                    axis=1,
                    inplace=True,
                )
            except:
                pass

            if sum(exl) > -1 :
                self.data = df
                self.data["Category"] = names
                continue

            df["Category"] = names

            self.dfs.append(df)

        

    def MergeScaleAndSplit(self):
        """_summary_
        """
        
       
        
        try:
            self.df
        except:
            self.fetchDfs()
            
        df_train = []
        df_val = []
        
        df_train_w = []
        df_val_w = []
        
        df_train_cat = []
        df_val_cat = []

        for df in self.dfs:
            x_b_train, x_b_val = train_test_split(df, test_size=0.2, random_state=seed)
            

            weights_train = x_b_train["wgt_SG"]
            weights_val = x_b_val["wgt_SG"]
            
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
        
        self.data_categories = self.data["Category"]
            
        self.data_weights = self.data["wgt_SG"]
        
        self.idx = np.where(X_b_val[X_b_val["Category"] == "singletop"])[0]
            
        channels = ["Zjets2",
            "diboson2L",
            "diboson3L",
            "diboson4L",
            "higgs",
            "singletop",
            "topOther",
            "Wjets",
            "triboson"
        ]
        
        idxs = []
        
        
        for channel in channels:
            
            idx_val = np.where(X_b_val["Category"] == channel)[0]
            idx_train = np.where(X_b_train["Category"] == channel)[0]
            idxs.append((idx_train, idx_val))
        
        

        X_b_train.drop("Category", axis=1, inplace=True)
        X_b_val.drop("Category", axis=1, inplace=True)
        self.data.drop("Category", axis=1, inplace=True)
        
        X_b_train.drop("wgt_SG", axis=1, inplace=True)
        X_b_val.drop("wgt_SG", axis=1, inplace=True)
        self.data.drop("wgt_SG", axis=1, inplace=True)

        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

        with strategy.scope():

            scaler_ae = MinMaxScaler()  # StandardScaler()#MinMaxScaler()
            self.X_b_train = scaler_ae.fit_transform(X_b_train)
            self.X_b_val = scaler_ae.transform(X_b_val)

            self.data = scaler_ae.transform(self.data)
            
        
        plotRMMMatrix = plotRMM(self.path, rmm_structure, 9)
        ### Plot RMM for each channel
        
        for idxss, channel in zip(idxs, channels):
            cols = X_b_train.columns
            idx_train, idx_val = idxss
            train = self.X_b_train[idx_train]
            val = self.X_b_val[idx_val]
            train = pd.DataFrame(data=train, columns=cols)
            val = pd.DataFrame(data=val, columns=cols)
            plot_df = pd.concat([train, val])
            plotRMMMatrix.plotDfRmmMatrix(plot_df, channel)

class RunAE:
    def __init__(self, data_structure, path:str):
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
        
        epochs = 5
        try:
            self.AE_model
        except:
            self.AE_model = self.getModel()

        with tf.device("/GPU:0"):
            tf.config.optimizer.set_jit("autoclustering")
            self.AE_model.fit(
                self.X_train,
                self.X_train,
                epochs=epochs,
                batch_size=8192,
                validation_split=0.2,
                sample_weight=self.data_structure.weights_train,
            )
            
        self.AE_model.save(f"tf_models/{epochs}_current_v.h5")
            
    def hyperParamSearch(self):
        """_summary_
        """
        
        device_lib.list_local_devices()
        tf.config.optimizer.set_jit("autoclustering")
        with tf.device("/GPU:0"):
            self.gridautoencoder(self.X_train, self.X_val)
            
    def gridautoencoder(self, X_b:np.ndarray, X_back_test:np.ndarray) -> None:
        """_summary_

        Args:
            X_b (np.ndarray): _description_
            X_back_test (np.ndarray): _description_
        """
        tuner = kt.Hyperband(
            self.AE_model_builder,
            objective=kt.Objective("val_mse", direction="min"),
            max_epochs=50,
            factor=3,
            directory="GridSearches",
            project_name="AE",
            overwrite=True,
        )
        print(tuner.search_space_summary())

        tuner.search(
            X_b, X_b, epochs=50, batch_size=8192, validation_data=(X_back_test, X_back_test), sample_weight=self.data_structure.weights_train
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
            answ = input("Do you want to save model? (y/n) ")
            if answ == "y":
                name = input("name: ")
                self.modelname = f"model_{name}.h5"
                tuner.hypermodel.build(best_hps).save("tf_models/"+self.modelname)
                state = False
                print(f"Model {self.modelname} saved")
                
                
            elif answ == "n":
                state = False
                print("Model not saved")


    def AE_model_builder(self, hp:kt.engine.hyperparameters.HyperParameters):
      
        """_summary_

        Args:
            hp (kt.engine.hyperparameters.HyperParameters): _description_

        Returns:
            _type_: _description_
        """
        ker_choice = hp.Choice("Kernel_reg", values=[0.5, 0.1, 0.05, 0.01])
        act_choice = hp.Choice("Atc_reg", values=[0.5, 0.1, 0.05, 0.01])

        alpha_choice = hp.Choice("alpha", values=[1.0, 0.5, 0.1, 0.05, 0.01])
        # get_custom_objects().update({"leakyrelu": tf.keras.layers.LeakyReLU(alpha=alpha_choice)})
        activations = {
            "relu": tf.nn.relu,
            "tanh": tf.nn.tanh,
            "leakyrelu": "leaky_relu",
            "linear": tf.keras.activations.linear,
        }#lambda x: tf.nn.leaky_relu(x, alpha=alpha_choice),
        inputs = tf.keras.layers.Input(shape=self.data_shape, name="encoder_input")
        x = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons1", min_value=60, max_value=self.data_shape - 1, step=1),
            activation=activations.get(
                hp.Choice("1_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(inputs)
        x_ = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons2", min_value=30, max_value=59, step=1),
            activation=activations.get(
                hp.Choice("2_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x)
        x1 = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons3", min_value=10, max_value=29, step=1),
            activation=activations.get(
                hp.Choice("3_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(x_)
        val = hp.Int("lat_num", min_value=1, max_value=9, step=1)
        x2 = tf.keras.layers.Dense(
            units=val,
            activation=activations.get(
                hp.Choice("4_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x1)
        encoder = tf.keras.Model(inputs, x2, name="encoder")

        latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")
        x = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons5", min_value=10, max_value=29, step=1),
            activation=activations.get(
                hp.Choice("5_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(latent_input)

        x_ = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons6", min_value=30, max_value=59, step=1),
            activation=activations.get(
                hp.Choice("6_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x)

        x1 = tf.keras.layers.Dense(
            units=hp.Int("num_of_neurons7", min_value=60, max_value=self.data_shape - 1, step=1),
            activation=activations.get(
                hp.Choice("7_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
            kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
            activity_regularizer=tf.keras.regularizers.L2(act_choice),
        )(x_)
        output = tf.keras.layers.Dense(
            self.data_shape,
            activation=activations.get(
                hp.Choice("8_act", ["relu", "tanh", "leakyrelu", "linear"])
            ),
        )(x1)
        decoder = tf.keras.Model(latent_input, output, name="decoder")

        outputs = decoder(encoder(inputs))
        AE_model = tf.keras.Model(inputs, outputs, name="AE_model")

        hp_learning_rate = hp.Choice("learning_rate", values=[9e-2, 9.5e-2, 1e-3, 1.5e-3])
        optimizer = tf.keras.optimizers.Adam(hp_learning_rate)
        # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        AE_model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

        return AE_model


    def runInference(self, tuned_model=False):
        """_summary_

        Args:
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
                        
                self.AE_model = tf.keras.models.load_model("tf_models/"+self.modelname+".h5")
            else:
                print("reg trained_model")
                self.AE_model = self.trainModel()

        with tf.device("/GPU:0"):
            self.pred_back = self.AE_model.predict(self.X_val, batch_size=8192)
            print("Background done")
            """
            pred_sig = self.AE_model.predict(test_set, batch_size=8192)
            print("Signal done")
            """

            self.pred_data = self.AE_model.predict(self.data, batch_size=8192)
            print("ATLAS data done")

        self.recon_err_back = self.reconstructionError(self.pred_back, self.X_val)
        self.recon_data = self.reconstructionError(self.pred_data, self.data)

    def reconstructionError(self, pred:np.ndarray, real:np.ndarray) -> np.ndarray:
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

    def checkReconError(self):
        """_summary_
        """
        
       

        # self.recon_err_back = self.recon_err_back.to_numpy()
        self.data_structure.weights_val = self.data_structure.weights_train.to_numpy()

        Zjets2 = self.recon_err_back[
            np.where(self.data_structure.val_categories == "Zjets2")[0]
        ]
        diboson2L = self.recon_err_back[
            np.where(self.data_structure.val_categories == "diboson2L")[0]
        ]
        diboson3L = self.recon_err_back[
            np.where(self.data_structure.val_categories == "diboson3L")[0]
        ]
        diboson4L = self.recon_err_back[
            np.where(self.data_structure.val_categories == "diboson4L")[0]
        ]
        triboson = self.recon_err_back[
            np.where(self.data_structure.val_categories == "triboson")[0]
        ]
        higgs = self.recon_err_back[
            np.where(self.data_structure.val_categories == "higgs")[0]
        ]
        singletop = self.recon_err_back[np.where(self.data_structure.val_categories == "singletop")[0]]
        
        topOther = self.recon_err_back[
            np.where(self.data_structure.val_categories == "topOther")[0]
        ]
        Wjets = self.recon_err_back[
            np.where(self.data_structure.val_categories == "Wjets")[0]
        ]
        ttbar = self.recon_err_back[
            np.where(self.data_structure.val_categories == "ttbar")[0]
        ]

        Zjets2_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "Zjets2")[0]
        ]
        diboson2L_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "diboson2L")[0]
        ]
        diboson3L_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "diboson3L")[0]
        ]
        diboson4L_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "diboson4L")[0]
        ]
        triboson_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "triboson")[0]
        ]
        higgs_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "higgs")[0]
        ]
        singletop_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "singletop")[0]
        ]
        topOther_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "topOther")[0]
        ]
        Wjets_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "Wjets")[0]
        ]
        ttbar_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "ttbar")[0]
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
            triboson,
        ] #ttbar,
        weight_atlas_data = [
            Zjets2_w,
            diboson2L_w,
            diboson3L_w,
            diboson4L_w,
            higgs_w,
            singletop_w,
            topOther_w,
            Wjets_w,
            triboson_w,
        ] #ttbar_w,
        
      

        sum_w = [np.sum(weight) for weight in weight_atlas_data]
        sort_w = np.argsort(sum_w, kind="mergesort")

   
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
            "triboson",
        ] #ttbar

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

        ax.scatter(x, N, marker="+", label="ttbar", color="black")

        ax.legend(prop={"size": 20})
        ax.set_title(
            "Reconstruction error histogram with background and ATLAS data", fontsize=25
        )
        ax.set_xlabel("Log10 Reconstruction Error", fontsize=25)
        ax.set_ylabel("#Events", fontsize=25)
        # ax.set_xlim([0, 3.5])
        ax.set_ylim([0.1, 5e6])
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=25)
        fig.tight_layout()
        plt.savefig(self.path + "/b_data_recon_big_rm3_feats.pdf")
        plt.close()
