import math
import random
import operator
import numpy as np
import mlmodel
import pandas as pd
import read_to_csv
from sklearn.cross_validation import train_test_split
from pyspark.sql import SparkSession
import time
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from scipy.stats import pearsonr
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

def make_dataframe(chromosome, train):
    position = np.array(zip(*np.where(chromosome == 1)))
    position[0,:] += 1
    position[1,:] += 1
    position = pd.DataFrame(position)
    train = pd.DataFrame(train)
    position.columns = ['position1','position2']
    train.columns = ['position1','position2','frequency','hitdirection','responsedirection','hitx','hity','hitz','responsex','reponsey','responsez','response']
    newtrain = pd.merge(train, position, on = ['position1','position2'], how = 'left', left_index = True)
    newtrain = newtrain[~pd.isnull(train.responsedirection)]
    return np.array(newtrain[['response', 'position1','position2','frequency','hitdirection','responsedirection','hitx','hity','hitz','responsex','reponsey','responsez']])

def runmodel_sklearn(chromosome, train, test, modelname):
    model = {'GBRT': GradientBoostingRegressor(max_depth = 7, loss = 'huber'),
             'xgb': xgb.XGBRegressor(objective='reg:linear', n_estimators = 10)}
    newtrain = make_dataframe(chromosome, train)
    estimator = model[modelname]
    return pearsonr(estimator.fit(newtrain[:,1:], newtrain[:,1]).predict(test[:,:-1]), test[:,-1])[0]

def evolve(spark, population, length, traindf, testdf, retain_rate = 0.2, random_select_rate = 0.5, mutation_rate = 0.01):
    parents = selection(spark, population, traindf, testdf, retain_rate, random_select_rate)
    population = crossover(parents, population, length)
    population = mutation(mutation_rate, population, length)
    return population, parents[0]
        
def gen_chromosome(length):
    return np.round(np.random.rand(length,length))
    
def gen_population(length, count, partitions):
    return [gen_chromosome(length) for i in xrange(count)]

def fitness(chromosome, traindf, testdf):
    #Score = mlmodel.runmodel_sklearn(chromosome, traindf, testdf, 'xgb')
    Score = runmodel_sklearn(chromosome, traindf, testdf, 'xgb')
    #Score = 1
    return Score * (chromosome.sum() + 1) ** -2
    
def my_sort(spark, population, traindf, testdf, partitions = 200):
    '''
    start = time.time()
    temp = spark.sparkContext.parallelize(population, partitions)
    GradedFitness = temp.map(lambda gen: fitness(gen, traindf, testdf)).collect() #use spark
    print('spark use %s s' % (time.time()-start))
    start = time.time()
    '''
    GradedFitness = [fitness(chromosome, traindf, testdf) for chromosome in population] #don't use spark
    #print('normal use %s s' % (time.time()-start))
    y = pd.DataFrame(GradedFitness)
    y.columns = ['fit']
    gradeindex = y.sort(['fit'],ascending=False)
    graded = [population[ind] for ind in gradeindex.index]
    return graded
        
def selection(spark, population, traindf, testdf, retain_rate, random_select_rate):
    graded = my_sort(spark, population, traindf, testdf)
    retain_length = int(len(graded) * retain_rate)
    parents = graded[:retain_length]
    for chromosome in graded[retain_length:]:
        if random.random() < random_select_rate:
            parents.append(chromosome)
    return parents
    
def crossover(parents, population, length):
    children = []
    target_count = len(population) - len(parents)
    while len(children) < target_count:
        male = random.randint(0, len(parents) - 1)
        female = random.randint(0, len(parents) - 1)
        if male != female:
            RowBegin = random.randint(0, length)
            RowEnd = random.randint(RowBegin, length)
            ColBegin = random.randint(0, length)
            ColEnd = random.randint(ColBegin, length)
            mask = np.zeros([length, length])
            for row in range(RowBegin, RowEnd):
                for col in range(ColBegin, ColEnd):
                    mask[row, col] = True
            male = parents[male]
            female = parents[female]
            child = (male.astype(bool) & mask.astype(bool)) + (female.astype(bool) & ~mask.astype(bool))
            children.append(child)
    population = parents + children
    return population
        
def mutation(rate, population, length):
    for i in xrange(len(population)):
        if random.random() < rate:
            Row = random.randint(0, length-1)
            Col = random.randint(0, length-1)
            population[i][Row, Col] = int(~population[i][Row, Col].astype(bool))
    return population

def result(best, traindf, testdf):
    return fitness(best, traindf, testdf)

def prepare_data(filename, feature):
    df = pd.read_csv(filename)
    usedfeature.append('response')
    return train_test_split(df[feature], df.response, test_size = 0.1, random_state = 1)

def genetic_algorithm(iteration_time, traindf, testdf):
    spark = SparkSession.builder.appName("Genetic Algorithm").getOrCreate()
    length = 95 #the diemension of the population matrix
    count = 50 #the population size
    partitions = 40 #the parallel partition number
    retain_rate = 0.2
    random_select_rate = 0.2
    mutation_rate = 0.08
    population = gen_population(length, count, partitions)
    for x in xrange(iteration_time):
        population, best = evolve(spark, population, length, traindf, testdf, retain_rate, random_select_rate, mutation_rate) #0.2, 0.1, 0.08
        if (x % 100) == 0:
            print 'round %s' %(x),
            print 'the score is %s' %(result(best, traindf, testdf))
    spark.stop()
    print 'the final score is %s' %(result(best, traindf, testdf))
    return best

if __name__ == '__main__':
    usedfeature = ['position1','position2','frequency','hitdirection','responsedirection','hitx','hity','hitz','responsex','reponsey','responsez']
    traindf, testdf, trainlabel, testlabel = prepare_data('FEMdata.csv', usedfeature)
    start = time.time()
    print genetic_algorithm(30000, traindf, testdf)
    print('spark use %s s' % (time.time()-start))

