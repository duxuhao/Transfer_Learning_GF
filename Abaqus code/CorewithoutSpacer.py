from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import numpy as np
import time
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup

def coordinate(mdb, nodenumber, modelname, instancename):
    fileid = open('FEMCoordinateless.csv','a')
    fileid.write('position1,x,y,z\n')
    for n in range(nodenumber):
        xyz = mdb.models[modelname].rootAssembly.instances[instancename].nodes[n].coordinates
        fileid.write(str(n+1))
        fileid.write(',')
        fileid.write(str(xyz[0]))
        fileid.write(',')
        fileid.write(str(xyz[1]))
        fileid.write(',')
        fileid.write(str(xyz[2]))
        fileid.write('\n')
    fileid.close()

start_time = time.time()
#executeOnCaeStartup()
#openMdb(pathName='D:/xuhao.du.UWA/Peter Du/transformer/UWATF/transfer_function/TFWithoutSpacer.cae')
#: The model database "D:\xuhao.du.UWA\Peter Du\transformer\UWATF\transfer_function\TFWithoutSpacer.cae" has been opened.
viewname = 'Viewport: 1'
modelname = 'Model-1'
instancename = 'Part-1-1'
loadname = 'Load-2'
jobsname = 'Job-1-e'
	
nodenumber = len(mdb.models[modelname].rootAssembly.instances[instancename].nodes)
#coordinate(mdb, nodenumber, modelname, instancename)
df = np.genfromtxt('FEMCoordinateless.csv', delimiter=',')
T = np.zeros(len(df)-1).astype(bool)
new = df[:,:]
new[1:,3] -= max(new[1:,3]/2)
'''
T |= ((np.round(np.abs(new[1:,1]),3) == 0.075) & (np.round(np.abs(new[1:,2]),3) <= 0.200))
T |= (np.round(np.abs(new[1:,3]),3) == 0.025)
T |= ((np.round(np.abs(new[1:,1]),3) == 0.125) & (np.round(np.abs(new[1:,3]),3) <= 0.025))
T |= ((np.round(np.abs(new[1:,2]),3) == 0.200) & (np.round(np.abs(new[1:,1]),3) <= 0.075))
T |= (np.round(np.abs(new[1:,2]),3) == 0.250)
'''
T = (np.round(np.abs(new[1:,1]),3) < 0.125) & (np.round(np.abs(new[1:,2]),3) < 0.250)& (np.round(np.abs(new[1:,3]),3) < 0.0250)
nodenumber = df[1:,0][T].astype(int)
print 'node quantity is {0}'.format(len(nodenumber))
a = mdb.models[modelname].rootAssembly
v1 = a.instances[instancename].vertices
nodenum = mdb.models[modelname].rootAssembly.instances[instancename].nodes.sequenceFromLabels(labels=nodenumber)
mdb.models[modelname].rootAssembly.Set(name='X', nodes=nodenum)

mdb.Job(name=jobsname, model=modelname, description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, numDomains=4, numGPUs=0)
recordfile = 'Response_100_solid_less.csv'
interupt = [0,0,0]

if sum(interupt) == 0:
    f = open(recordfile,'a')
    f.write('Hit_Position,Hit_Direction,Response_Position,Response_Direction,Response_100,Response_200\n')
    f.close()

ResponseDirection = ['A1', 'A2', 'A3']
for forceindex in xrange(len(ResponseDirection)):
    force = np.array([0,0,0])
    force[forceindex] = 1
    verts1 = v1.getSequenceFromMask(mask=('[#10 ]', ), )
    region = a.Set(vertices=verts1, name='Set-3')
    mdb.models[modelname].ConcentratedForce(name=loadname, createStepName='Step-2', region=region, cf1=force[0]+0j, cf2=force[1]+0j, cf3=force[2]+0j, distributionType=UNIFORM, field='', localCsys=None)
    #change the loading point n
    for n in nodenumber[interupt[forceindex]:]:
        print 'force direction %s at point %s' %(forceindex+1, n)
        nodenum = mdb.models[modelname].rootAssembly.instances[instancename].nodes.sequenceFromLabels(labels=(n,))
        mdb.models[modelname].rootAssembly.Set(name='Set-3', nodes=nodenum)
        mdb.jobs[jobsname].submit(consistencyChecking=OFF)
        mdb.jobs[jobsname].waitForCompletion() # wait for the job complete
        o1 = session.openOdb(name= jobsname + '.odb')
        session.viewports[viewname].setValues(displayedObject=o1)
        odb = session.odbs[jobsname + '.odb']
        f = open(recordfile,'a')
        for responseindex, d in enumerate(ResponseDirection):
            xyList = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('A', NODAL, ((COMPONENT, d), )), ), numericForm=REAL, nodeSets=('X', ))
            for i, num in enumerate(nodenumber):
                target1 = xyList[i][3][1]
                #target2 = xyList[i][-1][1]
                f.write(str(n) + ',' + str(forceindex) + ',' +  str(num) + ',' + str(responseindex) + ',' + str(target1) + '\n')
        for k in session.xyDataObjects.keys():
    		del session.xyDataObjects[k]
        f.close()
        odb.close()
        print time.time()-start_time
				
print 'total_time(s): ',
print time.time()-start_time