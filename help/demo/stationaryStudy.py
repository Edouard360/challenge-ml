# Following a tutorial we study here how our function of calls = f(t) is stationary
# We don't draw any conclusion from this...

from preprocessing.preprocessingClass import TrainPreprocessing
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pylab as plt

dataObj = TrainPreprocessing("../../preprocessing/output/trainPreprocessed.csv", ";")
data = dataObj.data[dataObj.data["ASS_ASSIGNMENT_Domicile"]==1].tail(5000)
data = data.reset_index()

data["CSPL_RECEIVED_CALLS"].plot()
data["CSPL_RECEIVED_CALLS"].rolling(center=True,window=336).mean().plot()

dftest = adfuller(data["CSPL_RECEIVED_CALLS"], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput) # Excellent p-value: we have a stationary evolution !


index_date = pd.date_range('1/1/2011', periods=5000, freq='1800s')
data = pd.DataFrame(data={"CSPL_RECEIVED_CALLS":data["CSPL_RECEIVED_CALLS"].values},index=index_date)
data['CSPL_RECEIVED_CALLS'] = data['CSPL_RECEIVED_CALLS'].astype('float64')

lag_acf = acf(data, nlags=20)
lag_pacf = pacf(data, nlags=20, method='ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(5000),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(5000),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(5000),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(5000),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')

plt.show()

