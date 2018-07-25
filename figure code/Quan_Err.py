import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('whitegrid')
df = pd.read_csv('Q_E.csv')
plt.plot(df.Quantity, 20 * np.log10(df.Error+1),'k')
plt.xscale('log')
plt.xlabel('Measured Quantity', fontsize=11)
plt.ylabel('RMSE-r (dB)', fontsize=11)
plt.savefig('Q_E',dpi = 600)
plt.show()