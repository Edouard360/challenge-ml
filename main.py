from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from regression.regressionClass import MultipleRegression,IndividualRegression

rgr = MultipleRegression("preprocessing/output/trainPreprocessedDateAssignment.csv","preprocessing/output/submissionPreprocessed.txt")

d = RandomForestRegressor(max_depth=11, random_state=2, n_estimators=30, criterion="linex")
d = BaggingRegressor(d, n_estimators = 40, max_samples=0.8,max_features=1.0)
d = DecisionTreeRegressor(max_depth=11,criterion='linex')

rgr.updateRegressor(d)
score = rgr.exportPredictionOnlyWithPastData("submission.txt")
