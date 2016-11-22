# This is work in progress (wip)
from preprocess.dataPreprocess import DataPreprocess,ResultPreprocess

ass = ["CMS","Crises","Domicile","Gestion","Gestion - Accueil Telephonique","Gestion Assurances","Gestion Relation Clienteles","Gestion Renault","Japon","Médical","Nuit","RENAULT","Regulation Medicale","SAP","Services","Tech. Axa","Tech. Inter","Téléphonie"]

path = '../../train_2011_2012_2013.csv'
#firstRow = 0 # The index of the first row to read
#lastRow = 0 # ~ Half the data (to make sure we have every rows in 2011)
dataObj = DataPreprocess(path, ";")# range(1, firstRow), lastRow - firstRow)
dataObj.preprocess(assignment = ass)
print(dataObj.data.columns)
dataObj.exportToCsv('../train/allPreprocessed.csv')

dataRes = ResultPreprocess('../../submission.txt', "\t")
dataRes.preprocess()
dataRes.exportToCsv('../submissionPreprocessed.txt')
dataRes.exportResult('../submission.txt')

