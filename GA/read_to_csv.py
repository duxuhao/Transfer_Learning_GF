import os
import pandas as pd
import numpy as np
from pylab import *
import array
import sys
from glob import glob

class datacontainer():
    def __init__(self, postfix):
        self.postfix = postfix
        
    def obtain_all_datfile(self, filelastname):
        direct = []
        for root, dirs, files in os.walk(".", topdown=False):
            for names in files:
                if filelastname in names:
                    direct.append(root + '/' + names)#obtain the file name and path
        return direct

    def read_file(self, filename):
        a = open(filename)
        fn = filename.split('/')
        position = fn[-1].split('_')
        for i in range(5):
            a.readline().strip()

        b = a.readline().strip()
        T = []
        while b != '':
            x = b.split(' ')
            T.append([position[1],position[2],position[3],position[4][0],x[0],x[-1]])
            b = a.readline().strip()
        a.close()
        x = pd.DataFrame(T)
        x.columns = ['position1','position2','hitdirection','responsedirection','frequency','response']
        return x

    def pickup_data(self):
        print '--- reading raw data ---'
        allfile = self.obtain_all_datfile(self.postfix)
        print 'total file %s' %(len(allfile))
        needfrequency = np.array([99.9534, 199.211, 324.241, 397.687, 498.636])
        for index, i in enumerate(allfile):
            if index == 0:
                x = self.read_file(i)
                
                T = x.frequency == needfrequency[0]
                for f in needfrequency:
                    T |= (x.frequency == str(f))
                x = x[T]
            else:
                df = self.read_file(i)
                x = pd.concat([x,df[T]])
        x.to_csv('FEMdata.csv', index = None)
        x = pd.read_csv('FEMdata.csv')
        coordinate = pd.read_csv(allfile[0].split('/')[1] + '/FEMCoordinate.csv')
        coordinate.columns = ['position1', 'hitx','hity','hitz']
        x = pd.merge(x, coordinate, on = 'position1', how = 'left', left_index = True)
        coordinate.columns = ['position2', 'responsex','reponsey','responsez']
        x = pd.merge(x, coordinate, on = 'position2', how = 'left', left_index = True)
        x.to_csv('FEMdata.csv', index = None)
        print '--- finsh exporting data ---'
