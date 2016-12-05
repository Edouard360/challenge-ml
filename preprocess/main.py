# Launch the script in the parent directory, ie doing:
# python preprocess/main.py (and not simply python main.py)
from preprocess.preprocessClass import DataPreprocess,ResultPreprocess
from preprocess.path import *

ass = ["CMS","Crises","Domicile","Gestion","Gestion - Accueil Telephonique","Gestion Assurances","Gestion Relation Clienteles","Gestion Renault","Japon","Médical","Nuit","RENAULT","Regulation Medicale","SAP","Services","Tech. Axa","Tech. Inter","Téléphonie"]

dataObj = DataPreprocess(PATH_TRAINING_DATA, ";",nrows=3000)
scaler = dataObj.preprocess(assignment = ass)
dataObj.exportToCsv("preprocess/output/trainPreprocessed.csv")

dataRes = ResultPreprocess(PATH_TEST_DATA, "\t",nrows=10000)
dataRes.preprocess(scaler)
dataRes.exportToCsv("preprocess/output/submissionPreprocessed.txt")
