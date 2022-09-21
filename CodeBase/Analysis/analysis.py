import numpy as np 
import pandas as pd 
from os import listdir
#import tensorflow as tf  
from pathlib import Path  
from typing import Tuple
import matplotlib.pyplot as plt
from os.path import isfile, join
#from sklearn import preprocessing 
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Analysis:
    def __init__(self, path:str) -> None:
        self.path = path
        self.onlyfiles = self.getDfNames()


    def getDfNames(self)-> Tuple[str, ...]:
        """
        Fetches all objects in a directory

        Returns:
            Tuple[str, ...]: list of pathnames
        """
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]


    def mergeDfs(self, exlude:list) -> None:
        files = self.onlyfiles.copy()
        

        dfs  = []
        for file in files:
            
            exl = [file.find(exl) for exl in exlude]
       
                
            if sum(exl) > -1:
                continue

            df = pd.read_hdf(self.path/file)
            dfs.append(df)


        self.df = pd.concat(dfs)

        print(len(self.df))


if __name__ == "__main__":
    #seed = tf.random.set_seed(1)
    path = Path("/storage/William_Sakarias/Sakarias_Data")

    exclude = ["data18"]

    analysis = Analysis(path)
    analysis.mergeDfs(exclude) 
    print(analysis.df["Channeltype"].unique())