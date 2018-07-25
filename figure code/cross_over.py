import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('whitegrid')

df1 = pd.read_csv('formal_swap_30.log')
df2= pd.read_csv('formal_swap_60.log')
df3 = pd.read_csv('formal_and_30.log')
df4 = pd.read_csv('formal_and_60.log')
df2 = df2.convert_objects(convert_numeric=True)
df4 = df4.convert_objects(convert_numeric=True)

plt.figure(figsize=(9,6))
plt.plot(df1.Round, np.log10(df1.MeasureNum),'ks-', label = 'Swap with 30 population',markersize=12)
plt.plot(df2.Round, np.log10(df2.MeasureNum),'ko-', label = 'Swap with 100 population',markersize=12)
plt.plot(df3.Round, np.log10(df3.MeasureNum),'k<-', label = 'And with 30 population',markersize=12)
plt.plot(df4.Round[:40], np.log10(df4.MeasureNum[:40]),'k*-', label = 'And with 100 population',markersize=12)
plt.xlim([0,10])
plt.ylim([1.5,6.5])
plt.legend(fontsize=12)
plt.xlabel('Interation Time', fontsize=14)
plt.ylabel('Quantity (log scale)', fontsize=14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.savefig('cross_over', dpi = 600)
#plt.show()