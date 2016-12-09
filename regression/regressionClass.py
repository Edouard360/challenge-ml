from preprocessing.preprocessingClass import TrainPreprocessing,SubmissionPreprocessing
from regression.score import linEx
from regression.tools import splitDummy,trainTestSplit
import pandas as pd
import abc

# It has been asked to predict only with the data of the past.
# This can be done using the MultipleRegression object,
# and calling the function exportPredictionOnlyWithPastData

predictionPeriods = [
('2012-12-28', '2013-01-04'),
('2013-02-02', '2013-02-09'),
('2013-03-06', '2013-03-13'),
('2013-04-10', '2013-04-17'),
('2013-05-13', '2013-05-20'),
('2013-06-12', '2013-06-19'),
('2013-07-16', '2013-07-23'),
('2013-08-15', '2013-08-22'),
('2013-09-14', '2013-09-21'),
('2013-10-18', '2013-10-25'),
('2013-11-20', '2013-11-27'),
('2013-12-22', '2013-12-29')]

class Regression(object):
    """
    The abstract Regression class.
    To manipulate test and train, perform and export regression results.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,pathPreprocessed,pathSubmission):
        self.dataObj = TrainPreprocessing(pathPreprocessed, ";")
        self.dataRes = SubmissionPreprocessing(pathSubmission, "\t")

        # By default, we want to test our data using a random split
        self.splitTestTrain()

        # The columns that will be used for the regression are retained as a attribute
        self.columns = self.dataObj.data.columns.difference(['ASS_ASSIGNMENT','JOUR_DE_L_AN', 'PAQUES','LUNDI_DE_PAQUES', 'FETE_DU_TRAVAIL', '8_MAI', 'ASCENSION','LUNDI_DE_PENTCOTE', 'FETE_NATIONALE', '15_AOUT', 'TOUSSAINT','11_NOVEMBRE', 'JOUR_DE_NOEL','MONTH','DATE_TIME','CSPL_RECEIVED_CALLS'])

    def updateRegressor(self, regressor):
        self.r = regressor

    def setTrainTestFromPath(self,pathTrain,pathTest):
        self.train = pd.read_csv(pathTrain, sep=";", header=0)
        self.test = pd.read_csv(pathTest, sep=";", header=0)

    def splitTestTrain(self,size = 0.1):
        self.train, self.test = trainTestSplit(self.dataObj.data, 0.1)

    def configForExport(self):
        self.train = self.dataObj.data
        self.test = self.dataRes.data

    def plotTest(self,assignment = "Téléphonie"):
        self.test = self.test.set_index(pd.DatetimeIndex(self.test["DATE_TIME"]))
        self.test.ix[self.test['ASS_ASSIGNMENT_'+assignment] == 1, :][["CSPL_RECEIVED_CALLS", "prediction"]].plot()

    def plotTrain(self,assignment = "Téléphonie"):
        self.train = self.train.set_index(pd.DatetimeIndex(self.train["DATE_TIME"]))
        self.train.ix[self.train['ASS_ASSIGNMENT_'+assignment] == 1, :]["CSPL_RECEIVED_CALLS"]

    @abc.abstractmethod
    def testOnTrainData(self):
        """
        Test the train data on a test set, which has been predefined beforehand
        By default the test data is taken from the initial train sample
        :param test_size:
        :return:
        """
        return

    @abc.abstractmethod
    def exportPrediction(self, path):
        """
        Export the result to the path of a regression fitted with all the train data
        using the self.r regressor.
        :param path:
        :return:
        """
        return

    @abc.abstractmethod
    def plotAssignment(self,assignment):
        """
        Plot the prediction and the real data obtained with testOnTrainData function
        :param assignment: A str giving the name of the assignment to be plotted
        :return:
        """
        return

class IndividualRegression(Regression):
    """
    The IndividualRegression class.
    It performs one regression per assignment
    """
    def testOnTrainData(self):
        self.test.insert(self.test.shape[1], "prediction", 0)
        self.test.is_copy = False
        self.regressionLoopOnAssignement(self.train, self.test)
        score = linEx(self.test["CSPL_RECEIVED_CALLS"].values, self.test["prediction"].values)
        print("The final score is :" + str(score))
        return score

    def exportPrediction(self,path):
        self.configForExport()
        self.regressionLoopOnAssignement(self.dataObj.data,self.dataRes.data)
        self.dataRes.exportResult(path)

    def regressionLoopOnAssignement(self,train,test):
        """
        The regressionLoopOnAssignement to loop over the assignment and do the individual regressions.
        :param train: a dataframe with the 'dummy coded' ASS_ASSIGNMENT features
        :param test: a dataframe with the 'dummy coded' ASS_ASSIGNMENT features.
        The test dataframe must contain a "prediction" column fo the function to fill.
        :return:
        """
        for train_, test_ in zip(splitDummy(train), splitDummy(test)):
            columns = train_.columns.intersection(self.columns)
            X_train = train_[columns].values
            y_train = train_["CSPL_RECEIVED_CALLS"].values
            X_test = test_[columns].values
            assert (X_train.shape[0] != 0), "Cannot predict one assignment"
            self.r.fit(X_train, y_train)
            self.test.loc[test_.index, "prediction"] = self.r.predict(X_test)

class MultipleRegression(Regression):
    """
    The MultipleRegression class.
    It performs only one regression using, ASS_ASSIGNMENT as a categorical feature.
    """
    def testOnTrainData(self):
        X_train = self.train[self.columns].values
        y_train = self.train["CSPL_RECEIVED_CALLS"].values
        X_test = self.test[self.columns].values
        y_test = self.test["CSPL_RECEIVED_CALLS"].values
        self.r.fit(X_train, y_train)
        score = linEx(y_test,self.r.predict(X_test))
        print("The final score is :" + str(score))
        return score

    def exportPrediction(self, path):
        self.configForExport()
        X_train = self.train[self.columns].values
        y_train = self.train["CSPL_RECEIVED_CALLS"].values
        X = self.test[self.columns].values
        self.r.fit(X_train, y_train)
        self.dataRes.data["prediction"] = self.r.predict(X)
        self.dataRes.exportResult(path)

    def exportPredictionOnlyWithPastData(self,path):
        """
        exportPredictionOnlyWithPastData is the only function that respect the requirements
        asked by the project to take into account only the data from the past, to predict the future
        :param path: the location where to export submission.txt
        :return:
        """
        train = self.dataObj.data.set_index(pd.Index(self.dataObj.data["DATE_TIME"]))
        test = self.dataRes.data.set_index(["DATE"])
        test.is_copy = False

        # For loop on all the weeks to predict
        for start,end in predictionPeriods:
            train_ = train.ix[:end]
            test_  = test[start:end]
            X_train = train_[self.columns].values
            y_train = train_["CSPL_RECEIVED_CALLS"].values
            X = test_[self.columns].values
            self.r.fit(X_train, y_train)
            test.loc[test_.index.unique(), "prediction"] = self.r.predict(X)
        self.dataRes.data["prediction"] = test["prediction"].values
        self.dataRes.exportResult(path)








