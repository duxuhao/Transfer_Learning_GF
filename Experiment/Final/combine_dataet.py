import pandas as pd

position = pd.read_csv('testarrange.csv')
df = pd.read_csv('testdataset.csv')
#x = df.groupby(['Num','Ch'])['Res'].mean().reset_index()

new = df.merge(position, on = ['Num','Ch'], how = 'left')
new.to_csv('completetest.csv', index = None)