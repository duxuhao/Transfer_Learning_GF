f = open('zdirection.csv','w')
f.write('Frequency')
for i in session.xyDataObjects.keys():
    f.write(',N{}'.format(i[-4:]))


f.write('\n')
for i in range(len(session.xyDataObjects[session.xyDataObjects.keys()[1]])):
    f.write('{}'.format(session.xyDataObjects[session.xyDataObjects.keys()[1]][i][0]))
    for j in session.xyDataObjects.keys():
        f.write(',{}'.format(session.xyDataObjects[j][i][1]))
    f.write('\n')


f.close()
