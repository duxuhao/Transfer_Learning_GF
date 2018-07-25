import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train.csv')
#print(np.unique(df.x_H))
#print(np.unique(df.y_H))
#print(np.unique(df.z_H))
#print(np.unique(df.x_R))
#print(np.unique(df.y_R))
#print(np.unique(df.z_R))
print(np.sum((df.Hit_Direction == 0) & (df.Response_Direction == 0) & (df.x_H == -0.125) & (df.z_H == 0.057)))
test = df[(df.Hit_Direction == 0) &
            (df.Response_Direction == 0) &
            (df.x_H == 0.125) &
            (df.y_H == 0.2) &
            (df.z_H == 0.057) &
            (df.y_R == 0.25) &
            (df.z_R == 0.057)
]
plt.plot(test.x_R, test.Response)
plt.show()
