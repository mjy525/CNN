import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""from pyramid.arima import auto_arima #auto ARIMA model (pip install pyramid-arima)"""
import xgboost as xgb #xgboost model

path = 'C:/Users/毛嘉宇/.spyder-py3/study/input/'
plt.rcParams["figure.figsize"] = [16,9]
train = open(path+'/train.csv')
train = pd.read_csv(train)
x = train[['date','store','item']]
y = train[['sales']]
test = open(path+'/test.csv')
test = pd.read_csv(test)
test.drop(['id'],axis=1,inplace=True)

x['date']       = pd.to_datetime(x['date'])
x['year']       = x['date'].dt.year
x['quarter']    = x['date'].dt.quarter
x['month']      = x['date'].dt.month
x['weekofyear'] = x['date'].dt.weekofyear
x['weekday']    = x['date'].dt.weekday
x['dayofweek']  = x['date'].dt.dayofweek
x.drop(['date'],axis=1,inplace=True)
x_train = x
x_test  = x
y_train = y
actual  = y

test['date']       = pd.to_datetime(test['date'])
test['year']       = test['date'].dt.year
test['quarter']    = test['date'].dt.quarter
test['month']      = test['date'].dt.month
test['weekofyear'] = test['date'].dt.weekofyear
test['weekday']    = test['date'].dt.weekday
test['dayofweek']  = test['date'].dt.dayofweek
test.drop(['date'],axis=1,inplace=True)


def SMAPE (forecast, actual):
    """Returns the Symmetric Mean Absolute Percentage Error between two Series"""
    forecast = forecast + 0.5
    forecast = forecast.astype(int)
    mse = np.array(forecast['sales'] - actual['sales'])
    acc = (len(mse[mse==0]) + len(mse[mse==1]) + len(mse[mse==-1]))/len(mse)
    print('SMAPE Error Score: ' + str(np.linalg.norm(mse,ord = 2)))
    print(acc)
    return mse,acc

        
def xboost(x_train, y_train, x_test):
    """Trains xgboost model and returns Series of predictions for x_test"""
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(x_train.columns))
    dtest = xgb.DMatrix(x_test, feature_names=list(x_test.columns))

    params = {'max_depth':10,
              'eta':0.8,
              'silent':0,
              'subsample':1.0}
    num_rounds = 100

    bst = xgb.train(params, dtrain, num_rounds)
    
    return pd.DataFrame(bst.predict(dtest),columns=['sales'])

forecast = xboost(x_train, y_train, x_test)
mse, acc = SMAPE(forecast, actual)
