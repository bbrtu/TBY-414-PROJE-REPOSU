#BARTU CAN OLCAY - 21698387
# ATIL AKGUŞ - 21893481


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from pandas import DataFrame
from matplotlib import pyplot
from math import sqrt




korona=pd.read_excel("data12.xlsx" , "data")
korona.head()
new_korona=korona.fillna(0)




korona_death = pd.DataFrame(korona.death)

korona_death['MA_4'] = korona.death.rolling(4).mean().shift()
korona_death['MA_7'] = korona.death.rolling(7).mean()
korona_death['MA_14'] = korona.death.rolling(14).mean()







plt.figure(figsize=(15,10))
plt.grid(True)
plt.plot(korona_death['death'],label='korona')
plt.plot(korona_death['MA_4'], label='Death Moving Avera 4 day')
plt.plot(korona_death['MA_7'], label='MA 7 day')
plt.plot(korona_death['MA_14'], label='MA 14 day')
plt.legend(loc=2)
plt.show()






def estimate_moving_average(new_korona,windowsize):
    avg = new_korona.rolling(windowsize).mean().iloc[-1]
    return avg
date= 4 
movingaverage = round(estimate_moving_average(new_korona,date),0)
print("Moving average estimation for last ", date, " date ", movingaverage)
print(movingaverage)

def estimate_moving_average(new_korona,windowsize):
    avg = new_korona.rolling(windowsize).mean().iloc[-1]
    return avg
date= 7 
movingaverage = round(estimate_moving_average(new_korona,date),0)
print("Moving average estimation for last ", date, " date ", movingaverage)
print(movingaverage)

def estimate_moving_average(new_korona,windowsize):
    avg = new_korona.rolling(windowsize).mean().iloc[-1]
    return avg
date= 14 
movingaverage = round(estimate_moving_average(new_korona,date),0)
print("Moving average estimation for last ", date, " date ", movingaverage)
print(movingaverage)




rolling_mean = korona_death.rolling(window = 12).mean()
rolling_std = korona_death.rolling(window = 12).std()

plt.plot(korona_death, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()

result = adfuller(new_korona['death'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
    
   
   
korona_log = np.log (korona_death) 
plt.plot (korona_log)   
plt.show()

rolling_mean = korona_log.rolling(window = 12).mean()
korona_log_minus_mean = korona_log - rolling_mean
korona_log_minus_mean.dropna(inplace=True)
plt.plot(korona_log_minus_mean)
plt.show()




korona_log_shift = korona_log - korona_log.shift()
korona_log_shift.dropna(inplace=True)
plt.plot(korona_log_shift)
plt.show()



model = ARIMA(new_korona.death, order=(1,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

model_fit.plot_predict(dynamic=False)
plt.show()




