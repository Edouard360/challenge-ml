import pandas as pd
import numpy as np
from itertools import chain

# 1 - Preprocessing step
# Load the data

firstRow = 1 # The index of the first row to read
lastRow = 1000 # The index of the last row to read
path = '../train_2011_2012_2013.csv' # The path to the csv file
indexInt = chain((1,3,6,7,8),range(18,84)) # Column numbers corresponding to int (in the csv file)
indexStr = (0,2,4,5) # Column numbers corresponding to string (in the csv file)
# Should normally be : chain((0,2,4,5),range(10,18)) / but pb with encoding in ascii

# Using pandas
# Type of the output : pandas.core.frame.DataFrame
# print("Data structure : ", type(data))
# print("Number of columns : ",len(data.columns))
# Note : data["TPER_TEAM"].values
# Note : data[["CSPL_RECEIVED_CALLS","CSPL_CALLS"]]
data = pd.read_csv(path,sep=";",skiprows=range(1, firstRow), nrows=lastRow-firstRow, header=0)

# Using numpy
# Type of the output : ndarray
# With usecols=[..] , we could isolate different types. (Not to mix strings with numbers)
data_np_int = np.genfromtxt(path,delimiter=";",max_rows=lastRow-firstRow, skip_header=firstRow,usecols=indexInt)
data_np_str = np.genfromtxt(path,delimiter=";",max_rows=lastRow-firstRow, skip_header=firstRow,usecols=indexStr,dtype=str)

# Insert the two columns "Jours" and "Nuit"
# We could simply do :
# data["Nuit"]= (data["TPER_TEAM"]=="Nuit")+0
# But, we would prefer to have the new columns at "TPER_TEAM" position.
index = int(np.where(data.columns == "TPER_TEAM")[0][0]) # Get the "TPER_TEAM" position
data.insert(index,"Jours",(data["TPER_TEAM"]=="Jours")+0)
data.insert(index,"Nuit",(data["TPER_TEAM"]=="Nuit")+0)
del data["TPER_TEAM"]

# Some first plots to work on

data["CSPL_RECEIVED_CALLS"].plot.hist(bins=5)

data["CSPL_RECEIVED_CALLS"].cumsum().plot()

data[["DAY_WE_DS","CSPL_RECEIVED_CALLS"]].groupby(["DAY_WE_DS"],sort=False).sum().plot.bar()


