# This file is a memo
X=[]
y=[]

# Features not to forget
"DAY_OFF"

# Iterate over array in parallel
array1=[];array2=[]
zip(array1,array2)

# Half the data is at row
firstRow = 0 # The index of the first row to read
lastRow = 5000000 # ~ Half the data (to make sure we have every rows in 2011)

# Logging
import logging
logging.basicConfig(filename="regression/config_result.log",level=logging.DEBUG)
logging.info("Test")

# Matrix Rank
import numpy as np
np.linalg.matrix_rank(X)

# _________________ #
#      sklearn      #
# _________________ #

# Using preprocessing
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()

# Splitting test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

# Using feature_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
skb = SelectKBest(chi2, k=2)
X_new = skb.fit_transform(X, y)
print(skb.scores_)

# Using LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(solver = "eigen",n_components=2)
Xs = clf.fit_transform(X, y)

# Using the gradient boosting regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
regr = DecisionTreeRegressor(max_depth=4)
clf = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, loss='huber', random_state=1)
clf.fit(X, y); clf.feature_importances_
fig, axs = plot_partial_dependence(clf, X, [0,1,(1,2),(2,3)], feature_names=['A','B'], n_jobs=3, grid_resolution=50)

# Doing cross validation
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=5, test_size=0.2,random_state=2)
list_train_test = [(X_train[train_index],X_train[test_index],y_train[train_index],y_train[test_index]) for (train_index, test_index) in ss.split(X_train)]
for X_train, X_test,y_train,y_test in list_train_test:
    linEx = []
    clf.fit(X_train,y_train)
    print(linEx(y_test,clf.predict(X_test)))

# Export graphiz
from sklearn.tree import export_graphviz
d = DecisionTreeRegressor(max_depth=3)
export_graphviz(d,out_file='tree.dot')

# _________________ #
#     DATAFRAME     #
# _________________ #

data=[]

# TPER_TEAM and DAY_WE_DS for categorical features
from preprocess.tools import createCategoricalFeatures
data = createCategoricalFeatures(data,'TPER_TEAM')
data = createCategoricalFeatures(data,'DAY_WE_DS')

# SELECT ONLY THOSE WITH EPOCH TIME
data = data.loc[lambda df: df.EPOCH <= 'epoch_time', :]

# Selecting
X = data.ix[:,(data.columns != 'ASS_ASSIGNMENT')&(data.columns != 'CSPL_RECEIVED_CALLS')].values

# Useful to replace columns
cols = list(data)
cols.insert(1, cols.pop(cols.index('ASS_ASSIGNMENT')))
data = data.ix[:, cols]

# Dropping columns : the object type, the na, and the useless ones
data = data.dropna(axis=1)
data = data.drop(data.select_dtypes(include=['object']).ix[:, 1:].columns, axis=1)
data = data.drop(['TPER_HOUR', 'SPLIT_COD', 'ACD_COD'], axis=1)

# Using pandas index
# .ix is the most general and will support any of the inputs in .loc and .iloc.

# Concat columns (axis = 1)
import pandas as pd
pd.concat([data.loc[:, "EPOCH":"DAY_OFF"], data.loc[:, "CSPL_RECEIVED_CALLS"]], axis=1)

# DataFrame with columns names
pd.DataFrame(columns=list('ABCD'))

# Aggregating with pandas
data.groupby(["WEEKDAY"]).agg({'CSPL_RECEIVED_CALLS' : lambda x: np.sum(x)/56.,'Jours' : np.mean})
data.groupby(["WEEKDAY"]).agg([np.mean,np.std])

#Export the scaler to csv file
scaler = preprocessing.StandardScaler()
pathScaler = "../preprocess/scaler.csv"
pd.DataFrame(data={'mean': scaler.mean_, 'var': scaler.var_}).to_csv(pathScaler,index=False)

# _________________ #
#       PLOT        #
# _________________ #

# Plot figure
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10)); plt.xlim((1, 12)); plt.ylim(0,5)

# Plot on same axis (shared x-axis)
colnames = []
f, axarray = plt.subplots(len(colnames), sharex=True)
for i in range(len(colnames)):
    axarray[i].plot()
    axarray[i].set_title()

# Scatter plotting
Xs = Xs[np.where( y == 1 )]
n = 5
plt.scatter(Xs[:n,0], Xs[:n,1],  c='b', alpha=0.1)

# Showfliers option in boxplot(showfliers=False)

