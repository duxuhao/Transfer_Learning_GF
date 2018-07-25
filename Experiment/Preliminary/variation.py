from function import *

def ReadFrequency(filename, frequency, Fs, Label):
    df = read_lvm(filename)
    Fs = 10000
    return tf(df[Label],df.D, Fs)[frequency-1]

fre = 50
Fs = 10000
sns.set_style("whitegrid")
g = 9.8
Cal = np.array([0.672055672044, 0.705873355384, 0.682634620835]) * g
T = np.zeros([4,5,3])
for j in range(1,6):
    for i,time in enumerate(range(30,50,5)):    
        filename = 'variation_test_FEM_Exp/'+str(time) +'_Pack' + str(j)
        for k,n in enumerate(['A','B','C']):
            T[i,j-1,k] = ReadFrequency(filename, fre, Fs, n) * Cal[k]
            plt.plot(ReadFrequency(filename, fre, Fs, n) * Cal[k])
            plt.show()

result = np.sum(T,axis = 0)
#print result
#plt.plot([result[0,2],result[0,2],result[0,2],result[0,2],result[0,2],result[0,2]])