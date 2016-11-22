# This folder is a memo help for functions I have come across that seem relevant to me


X=[]
y=[]

# Matrix Rank
import numpy as np
np.linalg.matrix_rank(X)

# Using preprocessing
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()

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

# Using the decision tree regressor

# Using the gradient boosting regressor ## TODO : check out the loss function
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
regr = DecisionTreeRegressor(max_depth=4)
clf = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, loss='huber', random_state=1)
clf.fit(X, y); clf.feature_importances_
fig, axs = plot_partial_dependence(clf, X, [0,1,(1,2),(2,3)], feature_names=['A','B'], n_jobs=3, grid_resolution=50)

# Scatter plotting
import matplotlib.pyplot as plt
Xs = Xs[np.where( y == 1 )]
n = 5
plt.scatter(Xs[:n,0], Xs[:n,1],  c='b', alpha=0.1)

# The dataframe
data=[]

# TPER_TEAM and DAY_WE_DS for categorical features
from preprocess.tools import createCategoricalFeatures
data = createCategoricalFeatures(data,'TPER_TEAM')
data = createCategoricalFeatures(data,'DAY_WE_DS')

#Selecting
X = data.ix[:,(data.columns != 'ASS_ASSIGNMENT')&(data.columns != 'CSPL_RECEIVED_CALLS')].values

# Useful to replace columns
cols = list(data)
cols.insert(1, cols.pop(cols.index('ASS_ASSIGNMENT')))
data = data.ix[:, cols]

# Dropping columns : the object type, the na, and the useless ones
data = data.drop(data.select_dtypes(include=['object']).ix[:, 1:].columns, axis=1)
data = data.dropna(axis=1)
data = data.drop(['TPER_HOUR', 'SPLIT_COD', 'ACD_COD'], axis=1)

# Plot on same axis (shared x-axis)
data = data.groupby(["START_OF_DAY","ASS_ASSIGNMENT"]).mean().reset_index()
colnames = []
f, axarray = plt.subplots(len(colnames), sharex=True)
for i in range(len(colnames)):
    axarray[i].plot()
    axarray[i].set_title()

# Features not to forget
"DAY_OFF"

# Using pandas index
# .ix is the most general and will support any of the inputs in .loc and .iloc.

# Concat columns (axis = 1)
import pandas as pd
pd.concat([data.loc[:, "EPOCH":"DAY_OFF"], data.loc[:, "CSPL_RECEIVED_CALLS"]], axis=1)

# DataFrame with columns names
pd.DataFrame(columns=list('ABCD'))

# Aggregating with pandas
data.groupby(["WEEKDAY"]).agg({'CSPL_RECEIVED_CALLS' : lambda x: np.sum(x)/56.,'Jours' : np.mean})