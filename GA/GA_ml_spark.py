from multiprocessing import Pool
from pyspark import SparkContext
import math
import random
import operator
import numpy as np
import mlmodel
import pandas as pd
import read_to_csv
from sklearn.cross_validation import train_test_split
import time

class GA():
    def __init__(self, spark, length, count):
        self.length = length
        self.count = count
        self.spark = spark
        self.population = self.gen_population(length, count)

    def evolve(self, traindf, testdf, retain_rate = 0.2, random_select_rate = 0.5, mutation_rate = 0.01):
        parents = self.selection(retain_rate, random_select_rate, traindf, testdf)
        self.crossover(parents)
        self.mutation(mutation_rate)
        
    def gen_chromosome(self, length):
        chromosome = np.round(np.random.rand(length,length))
        return chromosome
    
    def gen_population(self, length, count):
        return [self.gen_chromosome(length) for i in xrange(count)]
    
    def fitness(self, chromosome, traindf, testdf):
        MeasurementNum = chromosome.sum()
        start = time.time()
        Score = mlmodel.runmodel_sklearn(chromosome, traindf, testdf,'GBRT')
        Score = mlmodel.runmodel_spark(self.spark, chromosome, traindf, testdf,'GBRT')
        print('SPARK ml use %s s' % (time.time()-start))
        #print '%s measurement to score %s' %(MeasurementNum, Score)
        if Score > 0.24:
            return (MeasurementNum+1) ** -0.3
        else:
            return  Score * (MeasurementNum+1) ** -0.3
    
    def my_sort(self, spark, traindf, testdf):
        GradedFitness = [self.fitness(spark, chromosome, traindf, testdf) for chromosome in self.population]
        y = pd.DataFrame(GradedFitness)
        y.columns = ['fit']
        gradeindex = y.sort(['fit'],ascending=False)
        graded = [self.population[ind] for ind in gradeindex.index]
        return graded
        
    def selection(self, retain_rate, random_select_rate, traindf, testdf):
        graded = self.my_sort(traindf, testdf)
        retain_length = int(len(graded) * retain_rate)
        parents = graded[:retain_length]
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
                RowBegin = random.randint(0, self.length)
                RowEnd = random.randint(RowBegin, self.length)
                ColBegin = random.randint(0, self.length)
                ColEnd = random.randint(ColBegin, self.length)
                mask = np.zeros([self.length,self.length])
                for row in range(RowBegin, RowEnd):
                    for col in range(ColBegin, ColEnd):
                        mask[row, col] = True
                male = parents[male]
                female = parents[female]
                child = (male.astype(bool) & mask.astype(bool)) + (female.astype(bool) & ~mask.astype(bool))
                children.append(child)
        self.population = parents + children
        
    def mutation(self, rate):
        for i in xrange(len(self.population)):
            if random.random() < rate:
                Row = random.randint(0, self.length-1)
                Col = random.randint(0, self.length-1)
                self.population[i][Row, Col] = int(~self.population[i][Row, Col].astype(bool))
    
    def result(self, traindf, testdf):
        graded = self.my_sort(self.spark, traindf, testdf)
        print 'use %s transfer function to get' %(graded[0].astype(int).sum()),
        return self.fitness(graded[0],traindf, testdf)
    
if __name__ == '__main__':
    pool = Pool(4)
    #read_to_csv.datacontainer('20161114_').pickup_data()
    df = pd.read_csv('FEMdata.csv')
    usedfeature = ['position1','position2','frequency','hitdirection','responsedirection','hitx','hity','hitz','responsex','reponsey','responsez']
    usedfeature.append('response')
    train, test, trainlabel, testlabel = train_test_split(df[usedfeature], df.response, test_size = 0.1, random_state = 1)
    log = open('genetic.log1', 'a')
    log.write('log\n')
    log.close()
    sc = SparkContext(appName="Spark MLlib")
    ga = GA(sc, 95, 1000)
    log = open('genetic.log.spark', 'a')
    for x in xrange(5000):
        ga.evolve(train, test, retain_rate = 0.2, random_select_rate = 0.2, mutation_rate = 0.08) #0.2, 0.1, 0.08
        print 'the score is %s' %(ga.result(train, test))
        log.write('the score is ')
        log.write(str(ga.result(train, test)))
        log.write('\n')
        log.close()
