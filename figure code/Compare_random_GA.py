import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style("white")
df = pd.read_csv('Compare_random_GA.csv')
Fre = np.array([50,70,98,100,102,120,150])
plt.plot(Fre,20*np.log10(1+df.Q128), marker = 'o',label = 'Reduction Times: 5.0')
plt.plot(Fre,20*np.log10(1+df.Q148), marker = '*', label = 'Reduction Times: 4.3')
plt.plot(Fre,20*np.log10(1+df.Q289), marker = '^', label = 'Reduction Times: 2.2')
plt.plot(Fre,20*np.log10(1+df.Q321), marker = 's', label = 'Reduction Times: 2.0')
plt.xlabel('Frequency (Hz)')
plt.ylabel('RMSE-r (dB)')
plt.legend(loc = 2)
#plt.savefig('VariatewithFre.png',dpi=600)
plt.show()