from multiprocessing import Pool
import math
import time
import random
import operator
import numpy as np
import mlmodel
import pandas as pd
#import read_to_csv
from sklearn.cross_validation import train_test_split

def prepare_data(filename, feature, testSize = 0.3):
    df = pd.read_csv(filename)
    #T = ((np.round(np.abs(df.x_H),3) == 0.075) & (df.Hit_Direction == 0)) | ((np.round(np.abs(df.z_H),3) == 0.025) & (df.Hit_Direction == 2)) | ((np.round(np.abs(df.x_H),3) == 0.125) & (df.Hit_Direction == 0)) | ((np.round(np.abs(df.y_H),3) == 0.200) & (df.Hit_Direction == 1)) | ((np.round(np.abs(df.y_H),3) == 0.250) & (df.Hit_Direction == 1)) 
    position = pd.read_csv("FEMCoordinateless.csv")
    position.columns = ['Hit_Position','x_H','y_H','z_H']
    df = df.merge(position, on ='Hit_Position', how = 'left',left_index = True)
    Zmax = max(df.z_H)
    T = ((np.round(np.abs(df.x_H),3) == 0.075) & (df.Hit_Direction == 0)) | ((np.round(np.abs(df.z_H-Zmax/2.0),3) == 0.025) & (df.Hit_Direction == 2)) | ((np.round(np.abs(df.x_H),3) == 0.125) & (df.Hit_Direction == 0)) | ((np.round(np.abs(df.y_H),3) == 0.200) & (df.Hit_Direction == 1)) | ((np.round(np.abs(df.y_H),3) == 0.250) & (df.Hit_Direction == 1))
    df2 = df.copy()
    position.columns = ['Response_Position','x_R','y_R','z_R']
    df2 = df2.merge(position, on ='Response_Position', how = 'left',left_index = True)
    df = df[T]
    df = df.merge(position, on ='Response_Position', how = 'left',left_index = True)
    position = np.unique(df.Hit_Position)
    replacevalue = np.arange(len(position))
    #T.columns = ['Hit_Position','Response_Position']
    #train =df.merge(T, on = ['Hit_Position','Response_Position'], how = 'right')
    train = df.copy()
    train.Hit_Position.replace(position, replacevalue,inplace = True)
    train.Response_Position.replace(position, replacevalue,inplace = True)
    #train.to_csv('Train.csv',index = None)
    #T.columns = ['Response_Position', 'Hit_Position']
    #test =df.merge(T, on = ['Hit_Position','Response_Position'], how = 'right')
    #test.Hit_Position.replace(position, replacevalue,inplace = True)
    #test.Response_Position.replace(position, replacevalue,inplace = True)
    df2.Hit_Position.replace(position, replacevalue,inplace = True)
    df2.Response_Position.replace(position, replacevalue,inplace = True)
    return train, train.copy()


class GA():
    def __init__(self, length, count, modelname, usedfeature, label, former):
        self.length = length
        self.count = count
        self.modelname = modelname
        self.arraylen = length * length #length*(length-1)/2
        self.population = self.gen_population(former)
        print len(self.population)
        self.usedfeature = usedfeature
        self.label = label

    def evolve(self, traindf, testdf, retain_rate = 0.2, random_select_rate = 0.5, mutation_rate = 0.01):
        parents = self.selection(retain_rate, random_select_rate, traindf, testdf)
        self.crossover(parents)
        self.mutation(mutation_rate)
        
    def gen_chromosome(self):
        chromosome = np.round(np.random.rand(self.arraylen))
        return chromosome
    
    def gen_population(self, former):
        newborn = [self.gen_chromosome() for i in xrange(self.count)]
        f = open(former, 'r')
        a = f.readline().strip()
        while a != '':
            T = np.zeros(self.arraylen)
            for j in a.split(' ')[1:]:
                T[int(j)] = 1
            newborn.append(T)
            a = f.readline().strip()
        f.close()
        return newborn
    
    def score(self, Score, MeasurementNum):
        if Score < 0.6:
            return 1.0/ np.power(Score,1) /  (np.power(MeasurementNum,1)+1)
        else:
            return 0

    def fitness(self, chromosome, traindf, testdf, Score = 1):
        MeasurementNum = chromosome.sum()
        Score = mlmodel.runmodel_sklearn(chromosome, traindf, testdf, self.modelname,self.usedfeature,self.label)
        return  self.score(Score, MeasurementNum), Score, MeasurementNum
    
    def my_sort(self, traindf, testdf):
        GradedFitness = [self.fitness(chromosome, traindf, testdf)[0] for chromosome in self.population]
        y = pd.DataFrame(GradedFitness)
        y.columns = ['fit']
        gradeindex = y.sort_values(by = 'fit' ,ascending=False)
        graded = [self.population[ind] for ind in gradeindex.index]
        return graded
        
    def selection(self, retain_rate, random_select_rate, traindf, testdf):
        graded = self.my_sort(traindf, testdf)
        retain_length = int(len(graded) * retain_rate)
        parents = graded[:retain_length]
        self.best = parents[0].copy() #only using copy can present changing together
        for chromosome in graded[retain_length:]:
            if random.random() < random_select_rate:
                parents.append(chromosome)
        return parents
    
    def crossover(self, parents):
        children = []
        target_count = len(self.population) - len(parents)
        while len(children) < target_count:
            male = random.randint(0, len(parents) - 1)
            female = random.randint(0, len(parents) - 1)
            if male != female:
                #mask = np.round(np.random.rand(self.arraylen)).astype(bool)
                male = parents[male]
                female = parents[female]
                #child = (male.astype(bool) & mask) + (female.astype(bool) & ~mask)
                child = male.astype(bool) & female.astype(bool)
                children.append(child.astype(float))
        self.population = parents + children
        
    def mutation(self, rate):
        for i in xrange(len(self.population)):
            if (random.random() < rate) & (i != 0): # don't mute the best one
                self.population[i][random.randint(0, self.arraylen-1)] = 1 - self.population[i][random.randint(0, self.arraylen-1)]
    
    def print_result(self, traindf, testdf, string):
        print string + 'score: ',
        print self.fitness(self.best,traindf, testdf)

    def log_result(self, filename, traindf, testdf, round):
        a = open(filename,'a')
        evaluation = self.fitness(self.best,traindf, testdf)
        a.write('{0},{1},{2},{3},'.format(round, evaluation[0], evaluation[1], evaluation[2]))
        a.write(str(str(np.where(self.best == 1)[0])))
        a.write('\n')
        a.close()

if __name__ == '__main__':
    usedfeature = ['Hit_Direction','Response_Direction','x_H','y_H','z_H','x_R','y_R','z_R']
    train, test= prepare_data('Response_100_lessmesh.csv', usedfeature)
    #'''
    ga = GA(240, 100, 'ExtraTrees', usedfeature, 'Response', 'arrange.txt') # the population will influence the final performance even you have long iteration
    start_time = time.time()
    iteration = 5000
    filename = 'formal_threshold_06_Scoreplus_direction_noresiprosity_1_frf_add_former.log'
    f = open(filename,'a')
    f.write('Time,Round,Score,Model,MeasureNum,Arrange\n')
    f.close()
    for x in xrange(iteration):
        ga.evolve(train, test, retain_rate = 0.3, random_select_rate = 0.3, mutation_rate = 0.02) #0.2, 0.1, 0.08
        f = open(filename,'a')
        f.write(str(time.time()-start_time))
        f.write(',')
        f.close()
        ga.log_result(filename, train, test, x)
    print time.time()-start_time
    #'''
