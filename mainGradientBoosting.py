from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from preprocess.preprocessClass import DataPreprocess, ResultPreprocess
from regression.score import linEx
import logging
logging.basicConfig(filename="regression/config_result.log",level=logging.DEBUG)

# Loading the preprocessed train data
path = 'train/allPreprocessed.csv'
dataObj = DataPreprocess(path, ";")

# Loading the preprocessed result file
path = 'submissionPreprocessed.txt'
dataRes = ResultPreprocess(path, "\t")

# The dataframe should be indexed the same way
columns = dataObj.data.columns[(dataObj.data.columns != 'CSPL_RECEIVED_CALLS')]
X_train = dataObj.data[columns].values
y_train = dataObj.data["CSPL_RECEIVED_CALLS"].values
X = dataRes.data[columns].values

learning_rate = 0.6
n_estimators = 100
grad_factor = 100

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=2)

d = GradientBoostingRegressor(max_depth=3,
                              n_estimators=n_estimators,
                              learning_rate = learning_rate,
                              #loss='linex',
                              grad_factor=grad_factor,
                              #criterion = 'linex'
                              )

d.fit(X_train,y_train)

# # Regression on data X to predict
dataRes.data["prediction"] = d.predict(X)
dataRes.data["prediction"][dataRes.data["prediction"]<=0]=0
dataRes.exportResult("submission.txt")

# logging.info("The score for Gradient Boosting with depth "+str(d.max_depth)+" and n_estimators "+str(d.n_estimators)+" is :")
# logging.info(linEx(y_test,d.predict(X_test)))