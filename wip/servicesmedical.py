# This is work in progress (wip)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import matplotlib.pyplot as plt
from preprocess.dataPreprocess import DataPreprocess

from preprocess.tools import createTimeFeatures, createCategoricalFeatures

path = '../train/Services_Medical_2011.csv'
dataObj = DataPreprocess(path)

def exportMS():
    '''
    Procedure to obtain the 'train/Services_Medical_2011.csv' file
    from the 'train_2011_2012_2013.csv' file
    :return:
    '''
    path = '../../train_2011_2012_2013.csv'
    firstRow = 0  # The index of the first row to read
    lastRow = 5000000 # ~ Half the data (to make sure we have every rows in 2011)
    dataObj = DataPreprocess(path, range(1, firstRow), lastRow - firstRow)
    data = dataObj.data[dataObj.data['ASS_ASSIGNMENT'].apply(lambda x: x in ["Services", "Médical"])]
    data = createTimeFeatures(data)

    # EPOCH time corresponding to End 2011 beginning 2012
    data = data.loc[lambda df: df.EPOCH <= 1325370600000, :]

    data = createCategoricalFeatures(data,'TPER_TEAM')
    data = createCategoricalFeatures(data,'DAY_WE_DS')
    data = data.sort_values(by=["EPOCH","ASS_ASSIGNMENT"])

    cols = list(data)
    cols.insert(1, cols.pop(cols.index('ASS_ASSIGNMENT')))
    data = data.ix[:, cols]

    data = data.drop(data.select_dtypes(include=['object']).ix[:, 1:].columns, axis=1)
    data = data.dropna(axis=1)
    dataObj.data = data
    dataObj.exportToCsv('../train/Services_Medical_2011.csv')

def groupMS():
    '''
    Procedure to obtain the 'wip/Services_Medical_2011_grouped.csv' file
    from the 'train/Services_Medical_2011.csv' file

    Be careful when grouping features !
    Some need to be summed and others need to be averaged.
    :return:
    '''
    data = dataObj.data
    data = data.drop(['TPER_HOUR', 'SPLIT_COD', 'ACD_COD'],axis=1)
    data_tmp_sum = data.groupby(["EPOCH","ASS_ASSIGNMENT"]).sum()
    data_tmp_sum = data_tmp_sum.reset_index()
    data_tmp_mean = data.groupby(["EPOCH","ASS_ASSIGNMENT"]).mean()
    data_tmp_mean = data_tmp_mean.reset_index()

    list = ['START_OF_DAY','MONTH','WEEKDAY','DAY_OFF','WEEK_END','Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche','Jours','Nuit']
    for i in list: data_tmp_sum[i] = data_tmp_mean[i]
    dataObj.data = data_tmp_sum
    dataObj.exportToCsv('../train/Services_Medical_2011_grouped.csv')

def plotStartOfDayAveragedData():
    dataObj.data["START_OF_DAY"] = dataObj.data["START_OF_DAY"].astype('int')
    data = dataObj.data.groupby(["START_OF_DAY","ASS_ASSIGNMENT"]).mean()
    data = data.reset_index()
    dataMedical = data[data['ASS_ASSIGNMENT']=="Médical"]
    dataService = data[data['ASS_ASSIGNMENT'] == "Services"]
    colnames = ['CSPL_RECEIVED_CALLS', 'CSPL_ANSTIME', 'CSPL_HOLDCALLS', 'CSPL_HOLDTIME']
    f, axarray = plt.subplots(len(colnames), sharex=True)

    for i in range(len(colnames)):
        axarray[i].plot(dataService["START_OF_DAY"], dataService[colnames[i]])
        axarray[i].plot(dataMedical["START_OF_DAY"], dataMedical[colnames[i]])
        axarray[i].set_title(colnames[i])

def plotDayOfWeekAveragedData():
    dataObj.data["WEEKDAY"] = dataObj.data["WEEKDAY"].astype('int')
    data = dataObj.data.groupby(["WEEKDAY","ASS_ASSIGNMENT"]).mean()
    data = data.reset_index()
    dataMedical = data[data['ASS_ASSIGNMENT'] == "Médical"]
    dataService = data[data['ASS_ASSIGNMENT'] == "Services"]
    colnames = ['CSPL_RECEIVED_CALLS', 'CSPL_ANSTIME', 'CSPL_HOLDCALLS', 'CSPL_HOLDTIME']
    f, axarray = plt.subplots(len(colnames), sharex=True)

    for i in range(len(colnames)):
        axarray[i].plot(dataService["WEEKDAY"], dataService[colnames[i]],label = "Services")
        axarray[i].plot(dataMedical["WEEKDAY"], dataMedical[colnames[i]],label = "Médical")
        if(i==0):axarray[i].legend(loc="best",fontsize=12)
        axarray[i].set_title(colnames[i])

#plotDayOfWeekAveragedData()
#plotStartOfDayAveragedData()

data = dataObj.data
data["ASS_ASSIGNMENT"]=(data["ASS_ASSIGNMENT"]=="Services")+0

X = data.ix[:,(data.columns != 'ASS_ASSIGNMENT')&(data.columns != 'CSPL_RECEIVED_CALLS')].values
# X = data.ix[:, data.columns != 'ASS_ASSIGNMENT'].values
y_lda = data["ASS_ASSIGNMENT"].values

y = data["CSPL_RECEIVED_CALLS"].values
skb = SelectKBest(chi2, k=5)
X_new = skb.fit_transform(X, y)
print(skb.scores_)

#data = data.loc[lambda df: df.EPOCH <= 1325370600000, :]

clf = LinearDiscriminantAnalysis(solver = "eigen",n_components=2) # shrinkage and solver not supported
clf.fit(X_new, y_lda)
Xs = clf.transform(X_new)

Xs1= Xs[np.where( y_lda == 1 )]
Xs0= Xs[np.where( y_lda == 0 )]

np.random.shuffle(Xs1)
np.random.shuffle(Xs0)

n = 50000
plt.scatter(Xs0[:n,0], Xs0[:n,1],  c='b', alpha=0.1, label='Médical')
plt.scatter(Xs1[:n,0], Xs1[:n,1],  c='r', alpha=0.1, label='Services')
plt.legend(loc="best")
plt.title("LDA on ASS_ASSIGNMENT to reduce dimensions")
plt.show()

print("fiddle")