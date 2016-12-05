from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from regression.regressionClass import Regression

rgr = Regression("preprocessing/output/trainPreprocessed.csv","preprocessing/output/submissionPreprocessed.txt")

learning_rate = 0.4
n_estimators = 50
grad_factor = 10

d = GradientBoostingRegressor(max_depth=3, n_estimators=n_estimators,
                              learning_rate = learning_rate, grad_factor=grad_factor, loss='linex',random_state=2)

#d = RandomForestRegressor(max_depth=3,random_state=2,n_estimators = 5)#,criterion = "linex")

#d = DecisionTreeRegressor(max_depth=3,criterion='linex')

rgr.updateRegressor(d)
rgr.exportPredictionIndividual("submission.txt")
print("OK")