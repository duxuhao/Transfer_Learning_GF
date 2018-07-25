from multiprocessing import Pool
import math
import time
import random
import operator
import numpy as np
import FEMmlmodel
import pandas as pd
#import read_to_csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, Ridge, BayesianRidge, PassiveAggressiveRegressor, ElasticNet, Lars
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr


def prepare_data(filename, feature, testSize = 0.3):
    df = pd.read_csv(filename)
    position = pd.read_csv("FEMCoordinate.csv")
    position.columns = ['Hit_Position','x_H','y_H','z_H']
    df = df.merge(position, on ='Hit_Position', how = 'left',left_index = True)
    position.columns = ['Response_Position','x_R','y_R','z_R']
    df = df.merge(position, on ='Response_Position', how = 'left',left_index = True)
    position = np.unique(df.Hit_Position)
    replacevalue = np.arange(len(position))
    T = []
    for i in xrange(len(position)):
        for j in xrange(i+1,len(position)):
            T.append([position[i],position[j]])
    T = pd.DataFrame(T)
    T.columns = ['Hit_Position','Response_Position']
    train =df.merge(T, on = ['Hit_Position','Response_Position'], how = 'right')
    train.Hit_Position.replace(position, replacevalue,inplace = True)
    train.Response_Position.replace(position, replacevalue,inplace = True)
    T.columns = ['Response_Position', 'Hit_Position']
    test =df.merge(T, on = ['Hit_Position','Response_Position'], how = 'right')
    test.Hit_Position.replace(position, replacevalue,inplace = True)
    test.Response_Position.replace(position, replacevalue,inplace = True)
    return train, train.copy()

def make_dataframe(chromosome, train):
    b = chromosome
    position = np.zeros((len(b),2))
    position[:,0] = b
    position = pd.DataFrame(position.astype(int))
    position.columns = ['Hit_Position','delete']
    newtrain = train.merge(position, on = ['Hit_Position'], how = 'left')
    newtrain = newtrain[~pd.isnull(newtrain.delete)]
    return newtrain

def runmodel_sklearn(chromosome, train, test, modelname,feature,label):
    model = {'GBRT': GradientBoostingRegressor(max_depth = 7, loss = 'huber'),
             #'xgb': xgb.XGBRegressor(nthread = 10,objective='reg:linear', n_estimators = 10, max_depth = 3),
             'SVR': SVR(),
             'Lasso': Lasso(),
             'Linear': LinearRegression(),
             'DecisionTree':DecisionTreeRegressor(max_depth = 6),
             'RandomForest':RandomForestRegressor(random_state = 1, n_jobs=12),
             'Ridge':Ridge(),
             'AdaBoost':AdaBoostRegressor(),
             'BayesianRidge':BayesianRidge(compute_score=True),
             'KNN': KNeighborsRegressor(n_neighbors=12),
             'ExtraTrees': ExtraTreesRegressor(random_state = 1, n_jobs=12),
             'SGD': SGDRegressor(loss = 'huber', penalty = 'elasticnet', random_state = 1),
             'PassiveAggressive':PassiveAggressiveRegressor(),
             'ElasticNet': ElasticNet(),
             'Lars': Lars(),
             #'lgm': lgb.LGBMRegressor(objective='regression',num_leaves=40, learning_rate=0.1,n_estimators=20, num_threads = 10),
             #'xgb_parallel': xgb.XGBRegressor(objective='reg:linear', n_estimators = 10, max_depth = 3, nthread = 4)
            }

    newtrain = make_dataframe(chromosome, train)
    if len(newtrain) == 0:
        return 1000000000
    estimator = model[modelname]
    #return pearsonr(estimator.fit(newtrain[feature], newtrain[label]).predict(test[feature]), test[label])[0]
    estimator.fit(newtrain[feature], newtrain[label])
    return np.sqrt(np.power(estimator.predict(test[feature])-test[label],2).mean()) / np.sqrt(np.power(test[label],2).mean())

if __name__ == '__main__':
    usedfeature = ['Hit_Direction','Response_Direction','x_H','y_H','z_H','x_R','y_R','z_R']
    filename = ['Response2.csv','Response_98_102.csv','Response_70_120.csv','Response_50_150.csv','Response_40_170.csv','Response_80_130.csv']
    log = open('Arrange.log','a')
    log.write('Frequency,Quantity,R2\n')
    for fn in filename:
        print fn
        train, test= prepare_data(fn, usedfeature)
        for line in range(1,6):
            f = open('arrange.txt','r')
            for i in range(line):
                a = f.readline().strip().split(' ')
            f.close()
            chromosome = a[1:]
            print a[0]
            #chromosome = np.random.randint(0,635,len(chromosome))
            for label in ['Response','Response_98','Response_102']:
                try:
                    score = runmodel_sklearn(chromosome, train, test, 'ExtraTrees',usedfeature,label)
                    print label + ': ',
                    print score
                    if label == 'Response':
                        fre = 100
                    elif label == 'Response_98':
                        fre = int(fn.split('_')[1])
                    else:
                        fre = int(fn.split('_')[2][:3])
                    log.write('{0},{1},{2}\n'.format(fre,len(chromosome),score))
                except:
                    pass
    log.close()
