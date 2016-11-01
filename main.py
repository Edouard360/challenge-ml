import pandas as pd
import numpy as np
from itertools import chain
from matplotlib.pyplot import *
from preprocess import createTimeFeatures,createCategoricalFeatures, castToCategorialFeatures, normalize

# Load the data with 'numpy' or 'pandas' package

firstRow = 0 # The index of the first row to read
lastRow = 2000 # The index of the last row to read
path = '../train_2011_2012_2013.csv' # The path to the csv file
indexInt = chain((1,3,6,7,8),range(18,84)) # Column numbers corresponding to int (in the csv file)
indexStr = (0,2,4,5) # Column numbers corresponding to string (in the csv file)
# Should normally be : chain((0,2,4,5),range(10,18)) / but pb with encoding in ascii

# Using pandas
# Type of the output : pandas.core.frame.DataFrame
# print("Data structure : ", type(data))
# print("Number of columns : ",len(data.columns))
# print("Number of rows : ",len(data.index))
# Note : data["TPER_TEAM"].values
# Note : data[["CSPL_RECEIVED_CALLS","CSPL_CALLS"]]
data = pd.read_csv(path,sep=";",skiprows=range(1, firstRow), nrows=lastRow-firstRow, header=0)

# Using numpy
# Type of the output : ndarray
# With usecols=[..] , we could isolate different types. (Not to mix strings with numbers)
# data_np_int = np.genfromtxt(path,delimiter=";",max_rows=lastRow-firstRow, skip_header=firstRow,usecols=indexInt)
# data_np_str = np.genfromtxt(path,delimiter=";",max_rows=lastRow-firstRow, skip_header=firstRow,usecols=indexStr,dtype=str)

# 1 - Preprocessing step
data = castToCategorialFeatures(data)
data = createTimeFeatures(data)
data = createCategoricalFeatures(data,"TPER_TEAM")
data = normalize(data)

# Some first plots with normalized data

data[["DAY_WE_DS","CSPL_RECEIVED_CALLS"]].groupby(["DAY_WE_DS"],sort=False).mean().plot.bar()
xlabel('Days of Week')
ylabel('Received calls')

data[["TPER_HOUR","CSPL_RECEIVED_CALLS"]].groupby(["TPER_HOUR"],sort=False).mean().plot.bar()
xlabel('Hour of day')
ylabel('Received calls')

show()


