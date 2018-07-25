#import xgboost as xgb
#import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, Ridge, BayesianRidge, PassiveAggressiveRegressor, ElasticNet, Lars
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr
try:
    from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
    from pyspark.mllib.util import MLUtils
except:
    pass

def make_dataframe(chromosome, train):
    b = np.where(chromosome == 1)[0]
    #h = (np.floor((np.sqrt(b*8+1)+1)/2)-1).astype(int)+1
    #r = b - ((h ** 2 - h) / 2)
    h = (b/240).astype(int)
    r = b % 240
    position = np.zeros((len(b),3))
    position[:,0] = h
    position[:,1] = r
    position = pd.DataFrame(position.astype(int))
    position.columns = ['Response_Position','Hit_Position','delete']
    newtrain = train.merge(position, on = ['Hit_Position','Response_Position'], how = 'left')
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

def runmodel_spark(spark, train, test, modelname):
    newtrain = make_dataframe(chromosome, train)
    data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    model = GradientBoostedTrees.trainRegressor(trainingData,
                                                categoricalFeaturesInfo={}, numIterations=30)
