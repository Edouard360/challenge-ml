from preprocess.preprocessClass import DataPreprocess,ResultPreprocess
from regression.score import linEx
from regression.tools import splitDummy,split,trainTestSplit
from sklearn.model_selection import train_test_split

class Regression():
    def __init__(self,pathPreprocessed,pathSubmission):
        self.dataObj = DataPreprocess(pathPreprocessed, ";")
        self.dataRes = ResultPreprocess(pathSubmission, "\t")

    def updateRegressor(self, regressor):
        self.r = regressor

    def testOnTrainDataMultiple(self, test_size = 0.1):
        columns = self.dataObj.data.columns[(self.dataObj.data.columns != 'CSPL_RECEIVED_CALLS')]
        X_train = self.dataObj.data[columns].values
        y_train = self.dataObj.data["CSPL_RECEIVED_CALLS"].values
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=2)
        self.r.fit(X_train, y_train)
        score = linEx(y_test,self.r.predict(X_test))
        print("The final score is :"+str(score))

    def testOnTrainDataIndividual(self, test_size = 0.1):
        trainDf, testDf = trainTestSplit(self.dataObj.data,0.2)
        score = 0
        for train, res in zip(splitDummy(trainDf), splitDummy(testDf)):
            columns = train.columns[(train.columns != 'CSPL_RECEIVED_CALLS') & (train.columns != 'MONTH')]
            X_train = train[columns].values
            y_train = train["CSPL_RECEIVED_CALLS"].values
            X_test = res[columns].values
            self.r.fit(X_train, y_train)
            score += linEx(res["CSPL_RECEIVED_CALLS"].values, self.r.predict(X_test))
        print("The final score is :"+str(score))

    def exportPredictionMultiple(self,path):
        columns = self.dataObj.data.columns[(self.dataObj.data.columns != 'CSPL_RECEIVED_CALLS')]
        X_train = self.dataObj.data[columns].values
        y_train = self.dataObj.data["CSPL_RECEIVED_CALLS"].values
        X =  self.dataRes.data[columns].values
        self.r.fit(X_train, y_train)
        self.dataRes.data["prediction"] = self.r.predict(X)
        self.dataRes.exportResult(path)

    def exportPredictionIndividual(self,path):
        for train, res in zip(splitDummy(self.dataObj.data), split(self.dataRes.data)):
            columns = train.columns[(train.columns != 'CSPL_RECEIVED_CALLS') & (train.columns != 'MONTH')]
            X_train = train[columns].values
            y_train = train["CSPL_RECEIVED_CALLS"].values
            X = res[columns].values

            self.r.fit(X_train, y_train)
            self.dataRes.data.loc[res.index, "prediction"] = self.r.predict(X)
        self.dataRes.exportResult(path)