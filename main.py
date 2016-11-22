from sklearn.ensemble import GradientBoostingRegressor

from preprocess.dataPreprocess import DataPreprocess,ResultPreprocess
from sklearn.ensemble import RandomForestRegressor
from regression.score import linEx
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import logging

import numpy as np

logging.basicConfig(filename="config_result.log",level=logging.DEBUG)

ass = ["CMS","Crises","Domicile","Gestion","Gestion - Accueil Telephonique","Gestion Assurances","Gestion Relation Clienteles","Gestion Renault","Japon","Médical","Nuit","RENAULT","Regulation Medicale","SAP","Services","Tech. Axa","Tech. Inter","Téléphonie"]

path = 'train/allPreprocessed.csv'
dataObj = DataPreprocess(path, ";")

columns = dataObj.data.columns[dataObj.data.columns != 'CSPL_RECEIVED_CALLS']

y_train = dataObj.data["CSPL_RECEIVED_CALLS"].values
X_train = dataObj.data.ix[:,(dataObj.data.columns != 'CSPL_RECEIVED_CALLS')].values


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=2)

# n_components = 6
# pca = PCA(n_components=n_components)
# X_train = pca.fit_transform(X_train,y_train)
# X_test = pca.transform(X_test)

n_estimators = [100,300]
max_depth = [20,40,60]
#regr_rf = RandomForestRegressor(max_depth=m, random_state=2,n_estimators = n)
clf_array = [GradientBoostingRegressor(n_estimators=n, max_depth=d) for d in max_depth for n in n_estimators]

for r in clf_array:
    r.fit(X_train,y_train)
    y_pred = r.predict(X_test)
    #dataRes.data["prediction"]= regr_rf.predict(X_train)
    #dataRes.exportResult("submission2.txt")
    logging.info("The score for GBR with depth "+str(r.max_depth)+" and n_estimators "+ str(r.n_estimators)+" is:")
    logging.info(linEx(y_test,y_pred))#regr_rf.predict(X_test)))




#dataRes = ResultPreprocess('submissionPreprocessed.txt',"\t")
#X = dataRes.data.ix[:,columns].values
#assert dataRes.data.ix[:,columns].columns == columns




