from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from regression.regressionClass import Regression
import numpy as np
from numpy import power

rgr = Regression("../../preprocessing/output/trainPreprocessed.csv","preprocessing/output/submissionPreprocessed.txt")

rgr2 = Regression("../../preprocessing/output/trainPreprocessed.csv","preprocessing/output/submissionPreprocessed.txt")

array = [0.4,0.5,0.6,0.7,0.8,0.9,1]
linexComparisonFrame = pd.DataFrame(data={"power":array,"linex":np.zeros(len(array)),"pureLinex":np.zeros(len(array))})

for p,i in zip(array,range(len(array))):
    rgr2.dataObj.data["CSPL_RECEIVED_CALLS"] = power(rgr.dataObj.data["CSPL_RECEIVED_CALLS"],p)
    d = DecisionTreeRegressor(max_depth=1, criterion='linex')
    rgr2.updateRegressor(d)
    linexComparisonFrame.ix[i,"linex"]=rgr2.testOnTrainDataMultiple()
    d = DecisionTreeRegressor(max_depth=1, criterion='pureLinex')
    rgr2.updateRegressor(d)
    linexComparisonFrame.ix[i, "pureLinex"]=rgr2.testOnTrainDataMultiple()


linexComparisonFrame["pureLinex"] = linexComparisonFrame["pureLinex"]/linexComparisonFrame["linex"]
linexComparisonFrame["linex"] = linexComparisonFrame["linex"]/linexComparisonFrame["linex"]
linexComparisonFrame.columns = ['linex error', 'power', 'pureLinex error']
linexComparisonFrame.plot(x="power",title="linexError - Tree of depth 1")





