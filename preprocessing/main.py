# Launch the script in the parent directory, ie doing:
# python preprocessing/main.py (and not simply python main.py)
from preprocessing.preprocessingClass import TrainPreprocessing,SubmissionPreprocessing
from preprocessing.path import *

ass = ['CMS','Crises','Domicile','Gestion','Gestion - Accueil Telephonique','Gestion Assurances','Gestion Relation Clienteles','Gestion Renault','Japon','Médical','Nuit','RENAULT','Regulation Medicale','SAP','Services','Tech. Axa','Tech. Inter','Téléphonie','Tech. Total','Mécanicien','CAT','Manager','Gestion Clients','Gestion DZ','RTC','Prestataires']


dataObj = TrainPreprocessing(PATH_TRAINING_DATA, ";")
scaler = dataObj.preprocess(assignment = ass)
dataObj.exportToCsv("preprocessing/output/trainPreprocessed.csv")

dataRes = SubmissionPreprocessing(PATH_TEST_DATA, "\t")
dataRes.preprocess(scaler)
dataRes.exportToCsv("preprocessing/output/submissionPreprocessed.txt")
