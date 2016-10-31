import pandas as pd
import numpy as np

# 1 - Preprocessing step
# Load the data

firstRow = 1 # The index of the first row to read
lastRow = 1000 # The index of the last row to read
path = '../train_2011_2012_2013.csv' # The path to the csv file

# Using pandas
# Type of the output : pandas.core.frame.DataFrame
# print("Data structure : ", type(data))
# print("Number of columns : ",len(data.columns))
# Note : data["TPER_TEAM"].values
# Note : data[["CSPL_RECEIVED_CALLS","CSPL_CALLS"]]
data = pd.read_csv(path,sep=";",skiprows=range(1, firstRow), nrows=lastRow-firstRow, header=0)

# Using numpy
# Type of the output : ndarray
# dtype=[('myint','i8')]
# With usecols=[..] , we could isolate different types. (Not to mix strings with numbers)
data_np = np.genfromtxt(path,delimiter=";",max_rows=lastRow-firstRow, skip_header=firstRow)

# Insert the two columns "Jours" and "Nuit"
# We could simply do :
# data["Nuit"]= (data["TPER_TEAM"]=="Nuit")+0
# But, we would prefer to have the new columns at "TPER_TEAM" position.
index = int(np.where(data.columns == "TPER_TEAM")[0][0]) # Get the "TPER_TEAM" position

data.insert(index,"Jours",(data["TPER_TEAM"]=="Jours")+0)
data.insert(index,"Nuit",(data["TPER_TEAM"]=="Nuit")+0)
del data["TPER_TEAM"]
