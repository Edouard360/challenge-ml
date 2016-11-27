# This is work in progress (wip)
from preprocess.preprocess import DataPreprocess,ResultPreprocess

ass = ["CMS","Crises","Domicile","Gestion","Gestion - Accueil Telephonique","Gestion Assurances","Gestion Relation Clienteles","Gestion Renault","Japon","Médical","Nuit","RENAULT","Regulation Medicale","SAP","Services","Tech. Axa","Tech. Inter","Téléphonie"]

path = '../../train_2011_2012_2013.csv'

dataObj = DataPreprocess(path, ";")
scaler = dataObj.preprocess(assignment = ass)
dataObj.exportToCsv('../train/allPreprocessed.csv')

dataRes = ResultPreprocess('../../submission.txt', "\t")
dataRes.preprocess(scaler)
dataRes.exportToCsv('../submissionPreprocessed.txt')

