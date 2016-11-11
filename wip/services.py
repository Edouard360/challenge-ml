# This is work in progress (wip)
# This file gives you the first two plots obtained in graphs.

from preprocess.dataPreprocess import DataPreprocess
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from matplotlib.pyplot import *

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence

from preprocess.tools import createTimeFeatures, createCategoricalFeatures

path = 'train/Services_2011_weekly.csv'
dataObj = DataPreprocess(path)

def exportService():
    '''
    This function is just an example...
    Calling exportService creates the 'Services_2011_weekly.csv' file in the train folder.
    It corresponds to the number of calls averaged per weekday on a semi-hourly basis.
    :return:
    '''
    path = '../train_2011_2012_2013.csv'
    firstRow = 0  # The index of the first row to read
    lastRow = 5000000 # ~ Half the data (to make sure we have every rows in 2011)
    dataObj = DataPreprocess(path, range(1, firstRow), lastRow - firstRow)
    dataObj.data = dataObj.data[dataObj.data['ASS_ASSIGNMENT'] == "Services"]
    data = dataObj.data[['DATE','TPER_TEAM','DAY_WE_DS','CSPL_RECEIVED_CALLS']]
    data = createTimeFeatures(data,features_to_create=["EPOCH","START_OF_DAY","WEEKDAY"])

    # EPOCH time corresponding to End 2011 beginning 2012
    data = data.loc[lambda df: df.EPOCH <= 1325370600000, :]
    del data["EPOCH"]

    data = createCategoricalFeatures(data,'TPER_TEAM')
    data = createCategoricalFeatures(data,'DAY_WE_DS')
    data_tmp = data.groupby(["WEEKDAY","START_OF_DAY"]).agg(np.mean).reset_index()
    data_tmp[['CSPL_RECEIVED_CALLS']] = data.groupby(["WEEKDAY", "START_OF_DAY"]).agg({'CSPL_RECEIVED_CALLS': lambda x: np.sum(x) / 56.}).values
    dataObj.data = data_tmp[["START_OF_DAY","Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche","Nuit","Jours","CSPL_RECEIVED_CALLS"]]
    dataObj.exportToCsv("train/Services_2011_weekly.csv")

def regressionOnFeatureStartOfDay():
    '''
    Work in progress...
    Calling regressionOnFeatureStartOfDay plots the average number of calls on a semi-hourly basis,
    And show what the regression tree would produce with different depths.
    :return:
    '''

    data = dataObj.data.groupby('START_OF_DAY').mean()
    data = data.reset_index()
    X = data[['START_OF_DAY']].values
    y = data[['CSPL_RECEIVED_CALLS']].values

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    array_i =[1,3,5]

    trees_to_test = [(i,DecisionTreeRegressor(max_depth=i)) for i in array_i ]

    [t.fit(X, y) for (i,t) in trees_to_test]

    y_pred = [(i,t.predict(X)) for (i,t) in trees_to_test]

    plot(X,y,label="normal",linewidth=2.5)
    [plot(X,y,label="Depth: "+str(i)) for i,y in y_pred]
    legend(loc="best",fontsize=12)
    title("2011 - Services")
    ylabel("Average number of calls")
    xlabel("Time since start of day")
    show()
    return 0

def regressionOnTimeFeatures():
    '''
    Work in progress...
    Calling regressionOnTimeFeatures plots the partial dependence of the day features,
    according to a GradientBoostingRegressor,
    and show what the regression tree would produce with different depths.
    :return:
    '''
    data = dataObj.data
    X = data.ix[:, data.columns != 'CSPL_RECEIVED_CALLS'].values
    y = data[['CSPL_RECEIVED_CALLS']]

    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    clf.fit(X, y)
    print("Importance of features : ")
    print(clf.feature_importances_)

    features = [0,1,2,3,4,5,6,7,(7,6)]
    fig, axs = plot_partial_dependence(clf, X, features,
                                       feature_names=data.columns.values,
                                       n_jobs=3, grid_resolution=50)
    fig.suptitle('Partial dependence of CSPL_received_calls on time features\n'
                 'for Services_2011.csv')
    subplots_adjust(top=0.9)  # tight_layout causes overlap with subtitle

    show()
    return 0

regressionOnTimeFeatures()
