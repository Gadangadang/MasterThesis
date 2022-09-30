import matplotlib
import numpy as np
import pandas as pd
from os import listdir
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from os.path import isfile, join


class Data:
    def __init__(self, path: str) -> None:
        self.path = path
        self.onlyfiles = self.getDfNames()

    def getDfNames(self) -> Tuple[str, ...]:
        """
        Fetches all objects in a directory

        Returns:
            Tuple[str, ...]: list of pathnames
        """
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]

    def addColumnToDf(self, df_name: str, column_name: str, value: list) -> None:
        """
        Adds a column to the dataframe.
        if only single entry in vlaue, it is assumed that this value is for all rows in df.

        Args:
            df_name (str): name + filetype for the given dataframe file, must be .hdf5
            column_name (str): name of the column
            value (list): list of value or values.

        """

        try:
            df_name in self.onlyfiles

        except ValueError:
            print("Dataframe is not in directory, check for typos")

        df = pd.read_hdf(self.path / df_name)

        try:
            df[column_name]

        except:

            if len(value) != len(df):
                value = value * len(df)

            df[column_name] = value
            df.to_hdf(self.path / df_name, "mini")

    def printDf(self, df_name: str) -> None:
        """prints the first 5 entries in the dataframe, and the length of its columns

        Args:
            df_name (str): name of a given dataframe
        """

        df = pd.read_hdf(self.path / df_name)
        print(df.head(), len(df.columns))

    def addChannelType(self):
        """
        Adds channetype to the dataframe
        """

        column_name = "Channeltype"

        for idx, file in enumerate(self.onlyfiles):
            name_idx = file.find("_3lep")
            value = [f"{file[:name_idx]}"]

            self.addColumnToDf(df_name=file, column_name=column_name, value=value)
            self.printDf(file)


if __name__ == "__main__":
    path = Path("/storage/William_Sakarias/Sakarias_Data")

    data = Data(path)
    data.addChannelType()
