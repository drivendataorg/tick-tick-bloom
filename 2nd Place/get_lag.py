'''
This calculates the spatial lags
only in the past given input data
'''

from src import get_data

res_lag = get_data.get_spatiallag()
print(res_lag.describe().T)

res_lag.to_csv('./data/spat_lag.csv',index=False)