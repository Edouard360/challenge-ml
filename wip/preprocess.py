import pandas as pd
import numpy as np
from sklearn import preprocessing
from wip.tools import createTimeFeatures,createCategoricalFeatures, createDayNightFeature

def featureProcedure(data,delete=True):
    data = createTimeFeatures(data, ["EPOCH", "START_OF_DAY", "WEEKDAY","MONTH","HOLYDAYS"],delete)
    #data = createDayNightFeature(data)
    data = createCategoricalFeatures(data, "WEEKDAY",delete)
    data = createCategoricalFeatures(data, "HOLYDAYS",delete)
    #data = createCategoricalFeatures(data, "MONTH", delete)
    data = createCategoricalFeatures(data, "ASS_ASSIGNMENT",delete)
    return data

class Preprocess():
    def __init__(self, path, sep, skiprows=None, nrows=None, usecols=None):
        """
        For convenience, we will use pandas, as suggested by the project directives
        :param path: path string where to find data in csv format
        :param sep: string separating the data (";" or "\t")
        :param skiprows: list-like or integer, default None
        :param nrows: Number of rows of file to read. (Useful for reading pieces of large files)
        :param usecols: array-like, default None
        """
        self.sep = sep
        self.data = pd.read_csv(path, sep=self.sep, header=0, skiprows=skiprows, nrows=nrows, usecols=usecols)

    def exportToCsv(self,path):
        self.data.to_csv(path, sep=self.sep,index=False)

    def exportColumnsToCsv(self,path,columns):
        self.data[columns].to_csv(path, sep=self.sep, index=False,encoding='utf-8')

    def concatToCsv(self, path):
        with open(path, 'a') as f:
            self.data.to_csv(f, sep=self.sep,header=False, index=False)

class DataPreprocess(Preprocess):
    def preprocess(self, assignment = None):
        data = self.data
        if (assignment != None):
            data = data[data['ASS_ASSIGNMENT'].apply(lambda x: x in assignment)]
        data = data[['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']]
        data = data.groupby(["DATE", "ASS_ASSIGNMENT"]).sum()
        data = data.reset_index()
        data = featureProcedure(data)

        scaler = preprocessing.StandardScaler()
        scaler.fit(data[["EPOCH", "START_OF_DAY"]])
        data[["EPOCH", "START_OF_DAY"]] = scaler.transform(data[["EPOCH", "START_OF_DAY"]])
        data[["EPOCH", "START_OF_DAY"]] = round(data[["EPOCH", "START_OF_DAY"]], 4)

        self.data = data

        return scaler

class ResultPreprocess(Preprocess):
    def preprocess(self,scaler):
        self.data = featureProcedure(self.data,delete=False)
        self.data[["EPOCH", "START_OF_DAY"]] = scaler.transform(self.data[["EPOCH", "START_OF_DAY"]])
        self.data[["EPOCH", "START_OF_DAY"]] = round(self.data[["EPOCH", "START_OF_DAY"]], 4)

    def exportResult(self,path):
        self.exportColumnsToCsv(path,['DATE','ASS_ASSIGNMENT','prediction'])
