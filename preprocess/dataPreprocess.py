import pandas as pd
import numpy as np
from sklearn import preprocessing
from preprocess.tools import createTimeFeatures,createCategoricalFeatures, castToCategorialFeatures, normalize, normalizeWithMeanVariance

def featureProcedure(data,delete=True):
    data = createTimeFeatures(data, ["EPOCH", "START_OF_DAY", "WEEKDAY"],delete)
    data = createCategoricalFeatures(data, "WEEKDAY",delete)
    data = createCategoricalFeatures(data, "ASS_ASSIGNMENT",delete)
    return data

class Preprocess():
    def __init__(self, path, sep, skiprows=None, nrows=None, usecols=None):
        """
        We could load our data both using either pandas (read_csv) or numpy (genfromtxt).

        Using pandas,
        Type of output : pandas.core.frame.DataFrame (type(data))
        Valid operations : data.columns, data.index, data["TPER_TEAM"].values, data[["CSPL_RECEIVED_CALLS","CSPL_CALLS"]]

        Using numpy,
        Type of the output : ndarray
        As seen in previous TD, it is of best practice to differentiate between integers and strings
        indexInt = chain((1, 3, 6, 7, 8), range(18, 84))
        indexStr = chain((0,2,4,5),range(10,18)) (be careful with encoding in ascii)

        For convenience, we will use pandas, as suggested by the project directives

        :param path: path string where to find data in csv format
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

        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(data[["EPOCH", "START_OF_DAY"]])
        data[["EPOCH", "START_OF_DAY"]] = self.scaler.transform(data[["EPOCH", "START_OF_DAY"]])
        data[["EPOCH", "START_OF_DAY"]] = round(data[["EPOCH", "START_OF_DAY"]], 4)

        pathScaler = "../scaler.csv"
        pd.DataFrame(data={'mean': self.scaler.mean_, 'var': self.scaler.var_}).to_csv(pathScaler,index=False)

        data = data.loc[lambda df: df.EPOCH <= 1325370600000, :]
        self.data = data

class ResultPreprocess(Preprocess):
    def preprocess(self):
        self.data = featureProcedure(self.data,delete=False)
        pathScaler = "../scaler.csv"
        scalerInfo = pd.read_csv(pathScaler)
        self.data[["EPOCH", "START_OF_DAY"]] = (self.data[["EPOCH", "START_OF_DAY"]] - scalerInfo['mean'].values)/ np.sqrt(scalerInfo['var'].values)
        self.data[["EPOCH", "START_OF_DAY"]] = round(self.data[["EPOCH", "START_OF_DAY"]], 4)

    def exportResult(self,path):
        self.exportColumnsToCsv(path,['DATE','ASS_ASSIGNMENT','prediction'])

