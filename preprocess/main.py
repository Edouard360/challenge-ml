# Launch the script in the parent directory, ie doing:
# python preprocess/main.py (and not simply python main.py)
from preprocess.preprocessClass import DataPreprocess,ResultPreprocess
from preprocess.path import *

ass = ["CMS","Crises","Domicile","Gestion","Gestion - Accueil Telephonique","Gestion Assurances","Gestion Relation Clienteles","Gestion Renault","Japon","Médical","Nuit","RENAULT","Regulation Medicale","SAP","Services","Tech. Axa","Tech. Inter","Téléphonie"]

dataObj = DataPreprocess(PATH_TRAINING_DATA, ";")
scaler = dataObj.preprocess(assignment = ass)
dataObj.exportToCsv("output/trainPreprocessed2.csv")

dataRes = ResultPreprocess(PATH_TEST_DATA, "\t")
dataRes.preprocess(scaler)
dataRes.exportToCsv("output/submissionPreprocessed2.txt")
