import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('white')
x = pd.read_csv('femxdirection.csv')
z = pd.read_csv('femzdirection.csv')

xv = np.array(x.ix[:,1:])
zv = np.array(z.ix[:,1:])
xa = np.mean(xv, axis = 1)
za = np.mean(zv, axis = 1)

plt.plot(x.Frequency, 20*np.log10((xa+za)/2.0), 'k', label = 'FEM')
plt.show()