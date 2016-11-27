from sklearn.ensemble import GradientBoostingRegressor
from preprocess.preprocess import DataPreprocess, ResultPreprocess

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

learning_rate = 0.3
n_estimators = 130
grad_factor = 60

d = GradientBoostingRegressor(max_depth=3,
                              n_estimators=n_estimators,
                              learning_rate = learning_rate,
                              loss='linex',
                              grad_factor=grad_factor,
                              criterion='linex')
d.fit(X_train,y_train)

# Regression on data X to predict
dataRes.data["prediction"] = d.predict(X)
dataRes.data["prediction"][dataRes.data["prediction"]<=0]=0
dataRes.exportResult("submission.txt")
