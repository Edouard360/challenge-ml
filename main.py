import pandas as pd
import numpy as np
from matplotlib.pyplot import *

from preprocess.dataPreprocess import DataPreprocess

firstRow = 0 # The index of the first row to read
lastRow = 1000 # The index of the last row to read
usecols = ["DATE","CSPL_RECEIVED_CALLS"]
path = '../train_2011_2012_2013.csv' # The path to the csv file

dataObj = DataPreprocess(path,range(1, firstRow),lastRow-firstRow,usecols)

print(dataObj.data)




show()


