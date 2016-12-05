from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from regression.regressionClass import Regression
import numpy as np
from numpy import power
from regression.score import linEx

rgr = Regression("../../preprocessing/output/trainPreprocessed.csv","../../preprocessing/output/submissionPreprocessed.txt")

rgr2 = Regression("../../preprocessing/output/trainPreprocessed.csv","../../preprocessing/output/submissionPreprocessed.txt")

d = DecisionTreeRegressor(max_depth=1, criterion='linex')

rgr2.updateRegressor(d)

array = [0.4,0.6,0.8,1]
comparisonFrame = pd.DataFrame(data={"power":array,"linex power":np.zeros(len(array)),"linex times":np.zeros(len(array))})

for factor,i in zip(array,range(len(array))):
    rgr2.dataObj.data["CSPL_RECEIVED_CALLS"] = power(rgr.dataObj.data["CSPL_RECEIVED_CALLS"],factor)
    y_test2,pred2 = rgr2.testOnTrainDataMultiple()
    y_test2=power(y_test2,1/factor)
    pred2=power(pred2,1/factor)
    comparisonFrame.ix[i, "linex power"] = linEx(y_test2,pred2)
    rgr2.dataObj.data["CSPL_RECEIVED_CALLS"] = rgr.dataObj.data["CSPL_RECEIVED_CALLS"]* factor
    y_test2, pred2 = rgr2.testOnTrainDataMultiple()
    y_test2 = y_test2/ factor
    pred2 = pred2 / factor
    comparisonFrame.ix[i, "linex times"] = linEx(y_test2, pred2)

comparisonFrame.plot(x="power")