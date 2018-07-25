from function import *
from matplotlib import cm

def ReadFrequency(filename, frequency):
    df = read_lvm(filename)
    Fs = 10000
    return tf(df.A, df.B, Fs)#[frequency-1]

def PlotFile(filename):
    df = read_lvm(filename)
    Fs = 10000
    #plt.plot(20*np.log10(tf(df.A, df.B, Fs))[1:500],'k')
    
    #plt.savefig('Front', dpi = 600)
    #plt.show()
    #plt.close()
    return tf(df.A, df.B, Fs)[1:200]
    
def makemean(a):
    return a - np.mean(a)
    
def moedshape(f, df, value):
    plt.figure(figsize=(4.1,5.3))
    fig=plt.scatter(df.X, df.Y, c = value, s = 900, marker = 's', edgecolors = 'none',cmap=cm.PuBu) 
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-0.5,9.5])
    plt.axis('off')
    #plt.savefig('SideFrequency' + str(f) + ' Hz', dpi = 600)
    plt.show()

#sns.set_style("whitegrid")
#PlotFile('modeshapeexperiment/p28')
'''
xyz = pd.read_csv('Front.csv')
fre = 101
a = []
for i in range(5,13):
    filename = 'modeshapeexperiment/p' + str(i+1)
    #a.append(ReadFrequency(filename, fre))
    plt.plot(20*np.log10(ReadFrequency(filename, fre)),label = str(i))

plt.xlim([40,140])
plt.legend()
plt.show()
#moedshape(fre, xyz, a)
'''
#'''
a = np.zeros(199)
b = np.zeros(199)
x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,26,27,28,29,30,31,32,33,34,35,35]
for i in x[:23]:
    filename = 'modeshapeexperiment/p' + str(i+1)
    a += PlotFile(filename) ** 2

for i in x[23:]:
    filename = 'modeshapeexperiment/p' + str(i+1)
    b += PlotFile(filename) ** 2

ax = plt.subplot(111)
#plt.plot(np.arange(1,200),makemean(10*np.log10(a/23 + b / (len(x) - 23))),'k', label = 'Experiment')
plt.plot(np.arange(1,200),makemean(10*np.log10(b/23)),'k', label = 'Experiment')
plt.xlabel('Frequency (Hz)',fontsize=14)
plt.ylabel('Amplitude (dB)',fontsize=14)
#plt.savefig('frequency_response', dpi = 600)

#plot the fem result
#sns.set_style('white')
x = pd.read_csv('femxdirection.csv')
z = pd.read_csv('femzdirection.csv')
for i in range(1,x.shape[0]-1):
    if x.ix[i,1] * 2 / (x.ix[i-1,1] * x.ix[i+1,1]) < 0.05:
        x.ix[i,:] = x.ix[i-1,:]
for i in range(1,z.shape[0]-1):
    if z.ix[i,1] * 2 / (z.ix[i-1,1] * z.ix[i+1,1])< 0.05:
        z.ix[i,:] = z.ix[i-1,:]

xv = np.array(x.ix[:,1:])
zv = np.array(z.ix[:,1:])
xa = np.mean(xv, axis = 1)
za = np.mean(zv, axis = 1)

plt.plot(x.Frequency-1, 10*np.log10((xa+za)/2.0)+2, '--k', label = 'FEM')
plt.xlim([0,200])
plt.ylim([-20,20])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
ax.tick_params(direction='in')
#plt.savefig('frequency_response_fem_exp', dpi = 600)
plt.show()
#'''
