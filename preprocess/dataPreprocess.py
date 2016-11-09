import pandas as pd
from preprocess.tools import createTimeFeatures,createCategoricalFeatures, castToCategorialFeatures, normalize

class DataPreprocess():
    def __init__(self,path,skiprows=None,nrows=None,usecols=None):
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
        self.data = pd.read_csv(path, sep=";", header=0, skiprows=skiprows, nrows=nrows, usecols=usecols)

    def preprocess(self):
        self.data = castToCategorialFeatures(self.data)
        self.data = createCategoricalFeatures(self.data, "TPER_TEAM")
        self.data = createTimeFeatures(self.data)
        self.data = normalize(self.data)

    def exportToCsv(self,file_name):
        self.data.to_csv(file_name, sep=";",index=False)

    def concatToCsv(self, file_name):
        with open(file_name, 'a') as f:
            self.data.to_csv(f, sep=";",header=False, index=False)
