# This is work in progress (wip)
from wip.preprocess import DataPreprocess,ResultPreprocess
import constants as const

ass = ["CMS","Crises","Domicile","Gestion","Gestion - Accueil Telephonique","Gestion Assurances","Gestion Relation Clienteles","Gestion Renault","Japon","Médical","Nuit","RENAULT","Regulation Medicale","SAP","Services","Tech. Axa","Tech. Inter","Téléphonie"]

dataObj = DataPreprocess(const.PATH_TRAINING_DATA, ";", nrows = const.INPUT_DATA_NROWS)
scaler = dataObj.preprocess(assignment = ass)
dataObj.exportToCsv(const.PATH_TRAINING_DATA_PREPROCESSED)

dataRes = ResultPreprocess(const.PATH_TEST_DATA, "\t", nrows = const.OUTPUT_NROWS)
dataRes.preprocess(scaler)
dataRes.exportToCsv(const.PATH_TEST_DATA_PREPRPOCESSED)
