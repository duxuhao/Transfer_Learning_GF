import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

#sns.set_style('whitegrid')
df = pd.read_csv('Q_E.csv')
plt.plot(df.Quantity, 20 * np.log10(df.Error+1),'k')
plt.xscale('log')
plt.minorticks_on()
plt.grid(b=True, which='major', color=[0.5,0.5,0.5], linestyle='-')
plt.grid(b=True, which='minor', color=[0.3,0.3,0.3], linestyle='--')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Measured Quantity', fontsize=14)
plt.ylabel('RMSE-r (dB)', fontsize=14)
plt.savefig('Q_E',dpi = 600)
plt.show()
