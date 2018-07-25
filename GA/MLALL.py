#import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, Ridge, BayesianRidge, PassiveAggressiveRegressor, ElasticNet, Lars
from sklearn.neighbors import KNeighborsRegressor
from multiprocessing import Pool
import time

start = time.time()

#pool = Pool(10)
df = pd.read_csv('Response2.csv')
position = pd.read_csv("FEMCoordinate.csv")

position.columns = ['Hit_Position','x_H','y_H','z_H']
df = df.merge(position, on ='Hit_Position', how = 'left',left_index = True)
position.columns = ['Response_Position','x_R','y_R','z_R']
df = df.merge(position, on ='Response_Position', how = 'left',left_index = True)
'''
directionconvert = pd.DataFrame([[0,1,0,0],[1,0,1,0],[2,0,0,1]])
directionconvert.columns = ['Hit_Direction','HD1','HD2','HD3']
df = df.merge(directionconvert, on ='Hit_Direction', how = 'left',left_index = True)
directionconvert.columns = ['Response_Direction','RD1','RD2','RD3']
df = df.merge(directionconvert, on ='Response_Direction', how = 'left',left_index = True)
'''
feature = list(df.ix[:,df.columns != 'Response'].columns)
label = 'Response'
feature.remove('Hit_Position')
feature.remove('Response_Position')
X_train, X_test, y_train, y_test = train_test_split(df[feature], df[label], test_size = 0.9, random_state = 0)

#estimator = DecisionTreeRegressor(random_state = 1)
#estimator = RandomForestRegressor(random_state = 1,n_jobs=12)
estimator = ExtraTreesRegressor(random_state = 1,n_jobs=12)
#estimator = ElasticNet()
estimator.fit(X_train, y_train)
print estimator
Coorelationcoefficient = pearsonr(estimator.predict(X_test), y_test)[0]
rms= np.sqrt(np.mean((estimator.predict(X_test)-y_test)**2)) / np.sqrt(np.mean((y_test)**2))
rms= np.sqrt(np.mean(estimator.predict(X_test)**2)) / np.sqrt(np.mean((y_test)**2))
print time.time() - start
print 'Coorelation Coefficient: {0}\nRMS Ratio: {1}'.format(Coorelationcoefficient, rms)
print '-' * 50
print 'The feature importances are:'
for i,f in enumerate(feature):
    print f + ': {0}'.format(estimator.feature_importances_[i])
