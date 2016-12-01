from preprocess.preprocess import DataPreprocess
from sklearn.ensemble import RandomForestRegressor
from regression.score import linEx
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(filename="config_result.log",level=logging.DEBUG)

# Loading the preprocessed train data
path = 'train/allPreprocessed.csv'
dataObj = DataPreprocess(path, ";")

X_train = dataObj.data.ix[:,(dataObj.data.columns != 'CSPL_RECEIVED_CALLS')].values
y_train = dataObj.data["CSPL_RECEIVED_CALLS"].values

# Test on the training set! No result yet exported
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=2)

d = RandomForestRegressor(max_depth=20,random_state=2,n_estimators = 40)
d.fit(X_train,y_train)

logging.info("The score for Regression Tree with depth "+str(d.max_depth)+" and n_estimators "+str(d.n_estimators)+" is :")
logging.info(linEx(y_test,d.predict(X_test)))