import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import array
#import seaborn as sns
import sys
from multiprocessing import Pool
from glob import glob

def read_lvm(name):
    f = open(name, 'rb')
    a = 'start'
    T = []
    for i in range(24):
        a = f.readline()
    while a != '':
        T.append(a.split('\t'))
        a = f.readline().strip()
    value = pd.DataFrame(T[1:])
    value.columns = ['A','B','C','D','E','F','G','H']
    return value

def myfft(signal, fs):
    return abs(np.fft.rfft(signal, fs) / fs * 2)

def tf(a, b, fs):
    Fa = myfft(a, fs)
    Fb = myfft(b, fs)
    tfvalue = Fa / Fb
    return tfvalue

def obtain_all_datfile(filelastname):
	direct = []
	for root, dirs, files in os.walk(".", topdown=False):
		for names in files:
			if filelastname in names:
				direct.append(root + '/' + names)#obtain the file name and path
	return direct

def rename(filename):
    os.rename(filename, filename[:-1])
