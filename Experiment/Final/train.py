import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('completetrain.csv')
test = pd.read_csv('completetestmean.csv')
#test = test[(test.Res != 0)]# & (test.Num > 30)]
feature = ['Hit_Direction',	'Ch','x_H',	'y_H',	'z_H',	'x_R',	'y_R',	'z_R']
label = 'Res'
estimator = ExtraTreesRegressor(random_state = 0)
estimator.fit(df[feature],df[label])
#print estimator.score(test[feature],test[label])
a = np.sqrt(np.power(estimator.predict(test[feature])-test[label],2).mean()) / np.sqrt(np.power(test[label],2).mean())
predict = pd.read_csv('predict.csv')
predict.Ch = 1
predict.Hit_Direction = 1
plt.plot(estimator.predict(predict[feature]),label = 'right')
predict.x_R = -0.25
plt.plot(estimator.predict(predict[feature]),label = 'left')
plt.legend()
plt.title('{}: {} dB'.format(i, 20*np.log10(1+a)))
plt.show()