from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from scipy import stats

def scatter3d(ra, ar):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a=ax.scatter(ra.Frequency, ra.Quantity, 20*np.log10(1+ra.R2), c='b',s=40, marker='o')
    b=ax.scatter(ar.Frequency, ar.Quantity, 20*np.log10(1+ar.R2), c='r',s=40, marker='^')
    plt.legend([a,b],['random select','GA select'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Measured Quantity')
    ax.set_zlabel('RMSE-r (dB)')
    fig.tight_layout()
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    plt.savefig('Compare_Raondom_Arrange',dpi = 600)
    plt.show()
    print 'the difference between random and arrange is: {}'.format(np.mean(ar.R2-ra.R2) / np.mean(ar.R2))

def scatter2d(ra, ar):
    fig = plt.figure()
    ax = plt.subplot(111)
    #ar['Difference'] = 20*np.log10(1+ar.R2) - 20*np.log10(1+ra.R2)
    ar['Difference'] = (ra.R2 - ar.R2) / ra.R2
    #for i in np.unique(ar.Frequency):
        #plt.plot(ar[ar.Frequency == i].Quantity,ar[ar.Frequency == i].Difference,label='Frequency: {} Hz'.format(i))
    cc = ['k-*','k--o','k-.>','k:s','k--d']
    for index, i in enumerate(np.unique(ar.Quantity)):
        a1 = ar[ar.Quantity == i].Frequency
        a2 = ar[ar.Quantity == i].Difference
        plt.plot(a1,a2,cc[index],label='Selected Quantity: {}'.format(i))
    #ax = fig.add_subplot(111, projection='3d')
    #a=ax.scatter(ra.Frequency, ra.Quantity, 20*np.log10(1+ra.R2), c='b',s=40, marker='o')
    #b=ax.scatter(ar.Frequency, ar.Quantity, 20*np.log10(1+ar.R2), c='r',s=40, marker='^')
    #plt.legend([a,b],['random select','GA select'])
#    plt.xticks([])
    ax.set_xlabel('Frequency (Hz)',fontsize = 14)
    #ax.set_xlabel('Measured Quantity')
    ax.set_ylabel('Relative Improve',fontsize=14)
    ax.tick_params(direction='in')
    fig.tight_layout()
    plt.legend(fontsize = 10)
    print(ar.Difference.mean())
    #plt.savefig('Compare_Raondom_Arrange_2d',dpi = 600)
    plt.show()

def bar2d(ra, ar):
    fig = plt.figure(figsize = (12,6))
    ax = plt.subplot(111)
    #ar['Difference'] = 20*np.log10(1+ar.R2) - 20*np.log10(1+ra.R2)
    ar['Difference'] = (ra.R2 - ar.R2) / ra.R2
    ar = ar[ar.Quantity != 154].reset_index(drop=True)
    #for i in np.unique(ar.Frequency):
        #plt.plot(ar[ar.Frequency == i].Quantity,ar[ar.Frequency == i].Difference,label='Frequency: {} Hz'.format(i))
    cc = ['k-*','k--o','k-.>','k:s','k--d']
    cc = ['k','r','b','y']
    for index, i in enumerate(np.unique(ar.Quantity)):
        a1 = ar[ar.Quantity == i].Frequency.values
        a2 = ar[ar.Quantity == i].Difference.values * 100
#        print(a1)
        plt.bar(np.arange(len(a1)) * 10 +1.5*index-4 * 1.5 / 2,a2,1,color = cc[index], label='Selected Quantity: {}'.format(i))
    #ax = fig.add_subplot(111, projection='3d')
    #a=ax.scatter(ra.Frequency, ra.Quantity, 20*np.log10(1+ra.R2), c='b',s=40, marker='o')
    #b=ax.scatter(ar.Frequency, ar.Quantity, 20*np.log10(1+ar.R2), c='r',s=40, marker='^')
    #plt.legend([a,b],['random select','GA select'])
    ax.set_xlabel('Frequency (Hz)',fontsize = 14)
    #ax.set_xlabel('Measured Quantity')
    ax.set_ylabel('Relative Improvement (%)',fontsize=14)
    ax.tick_params(direction='in')
    plt.xticks(range(0,115,10),[ 40,  50,  70,  80,  98, 100, 102, 120, 130, 150, 170])
    fig.tight_layout()
    plt.legend(fontsize = 14)
    print(ar.Difference.mean())
    print(ar[ar.Frequency == 100].Difference.mean())
    plt.savefig('Compare_Raondom_Arrange_2d',dpi = 600)
    plt.show()

def fitting(ra, ar):
#    fig = plt.figure(figsize = (12,6))
    fre = np.array([30,50,70,80,98,100,102,120,130,150,170])
    f=np.arange(50,151,50)
    #print 20*np.log10(1+ar[ar.Quantity == 128].R2[5:6].values[0]) #/ 20*np.log10(1+ar[ar.Quantity == 128].R2[5:6])
    print stats.linregress(fre[2:-1], ar[ar.Quantity == 128].R2[2:-1]/ar[ar.Quantity == 128].R2[5:6].values[0])
    print stats.linregress(fre[2:-1], ar[ar.Quantity == 148].R2[2:-1]/ar[ar.Quantity == 148].R2[5:6].values[0])
    print stats.linregress(fre[2:-1], ar[ar.Quantity == 289].R2[2:-1]/ar[ar.Quantity == 289].R2[5:6].values[0])
    print stats.linregress(fre[2:-1], ar[ar.Quantity == 321].R2[2:-1]/ar[ar.Quantity == 321].R2[5:6].values[0])
#    fig = plt.figure()
    fig = plt.figure(figsize = (12,6))
    ax = plt.subplot(111)
    print(ar[ar.Quantity == 128].R2[5:6])
    plt.plot(f,ar[ar.Quantity == 128].R2[5:6].values[0]*0+f/100.0,'k--',label='Error Curve')
#    plt.plot(f,ar[ar.Quantity == 128].R2[5:6].values[0]*0+f/100.0,'k--o',label='Selected Quantity: 128')
#    plt.plot(f,ar[ar.Quantity == 148].R2[5:6].values[0]*0+f/100.0,'r-.^',label='Selected Quantity: 148')
#    plt.plot(f,ar[ar.Quantity == 289].R2[5:6].values[0]*0+f/100.0,'b:*',label='Selected Quantity: 289')
#    plt.plot(f,ar[ar.Quantity == 321].R2[5:6].values[0]*0+f/100.0,'y-s',label='Selected Quantity: 321')
#    plt.plot(f,20*np.log10(1+1.*(ar[ar.Quantity == 128].R2[5:6].values[0])*f/100.0)-0.00,'k--o',label='Selected Quantity: 128')
#    plt.plot(f,20*np.log10(1+1.*(ar[ar.Quantity == 148].R2[5:6].values[0])*f/100.0)-0.00,'r-.^',label='Selected Quantity: 148')
#    plt.plot(f,20*np.log10(1+1.*(ar[ar.Quantity == 289].R2[5:6].values[0])*f/100.0)-0.00,'b:*',label='Selected Quantity: 289')
#    plt.plot(f,20*np.log10(1+1.*(ar[ar.Quantity == 321].R2[5:6].values[0])*f/100.0)-0.00,'y-s',label='Selected Quantity: 321')
#    plt.legend(loc = 2,fontsize = 14)
    plt.scatter(fre[1:-1], (ar[ar.Quantity == 128].R2[1:-1]/ar[ar.Quantity == 128].R2[5:6].values[0]),c='k',s=50, marker='o',label='Selected Quantity: 128')
    plt.scatter(fre[1:-1], (ar[ar.Quantity == 148].R2[1:-1]/ar[ar.Quantity == 148].R2[5:6].values[0]),c='r',s=50, marker='^',label='Selected Quantity: 148')
    plt.scatter(fre[1:-1], (ar[ar.Quantity == 289].R2[1:-1]/ar[ar.Quantity == 289].R2[5:6].values[0]),c='b',s=50, marker='*',label='Selected Quantity: 289')
    plt.scatter(fre[1:-1], (ar[ar.Quantity == 321].R2[1:-1]/ar[ar.Quantity == 321].R2[5:6].values[0]),c='y',s=50, marker='s',label='Selected Quantity: 321')
    plt.legend(loc = 2,fontsize = 14)
    for i in [128,148,289,321]:
        print(stats.pearsonr(fre[1:-1]/100.0, ar[ar.Quantity == i].R2[1:-1]/ar[ar.Quantity == i].R2[5:6].values[0]))
#    plt.scatter(fre[1:-1], 20*np.log10(1+ar[ar.Quantity == 128].R2[1:-1]),c='k',s=40, marker='o')
#    plt.scatter(fre[1:-1], 20*np.log10(1+ar[ar.Quantity == 148].R2[1:-1]),c='r',s=40, marker='^')
#    plt.scatter(fre[1:-1], 20*np.log10(1+ar[ar.Quantity == 289].R2[1:-1]),c='b',s=40, marker='*')
#    plt.scatter(fre[1:-1], 20*np.log10(1+ar[ar.Quantity == 321].R2[1:-1]),c='y',s=40, marker='s')
    #plt.plot(f,20*np.log10(1+1.*(ar[ar.Quantity == 128].R2[5:6].values[0])*f/100.0)-0.00,'k--o',markersize=2,label='Quantity Reduction Time: 5.0')
    #plt.plot(f,20*np.log10(1+1.*(ar[ar.Quantity == 148].R2[5:6].values[0])*f/100.0)-0.00,'k-.^',markersize=2,label='Quantity Reduction Time: 4.3')
    #plt.plot(f,20*np.log10(1+1.*(ar[ar.Quantity == 289].R2[5:6].values[0])*f/100.0)-0.00,'k:*',markersize=2,label='Quantity Reduction Time: 2.2')
    #plt.plot(f,20*np.log10(1+1.*(ar[ar.Quantity == 321].R2[5:6].values[0])*f/100.0)-0.00,'k-s',markersize=2,label='Quantity Reduction Time: 2.0')
    #plt.legend(loc = 2,fontsize = 10)
#    plt.xticks(fontsize=14)
#    plt.yticks(fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize = 14)
    plt.ylabel(r'$\frac{f_e}{f_s}$' + '    \     '+ r'$\frac{r_e}{r_s}$',fontsize = 20)
    ax.tick_params(direction='in')
    fig.tight_layout()
    plt.savefig('VariatewithFre',dpi = 600)
    plt.show()
    
ar = pd.read_csv('Arrange.log')
ra = pd.read_csv('Random1.log')
ar.sort_values(['Frequency','Quantity'],inplace=True)
#sns.set_style("white")
#scatter2d(ra, ar)
#fitting(ra, ar)
bar2d(ra,ar)
