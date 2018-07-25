from __future__ import print_function
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
from random import random
from operator import add
from multiprocessing import Pool
import time
import numpy as np
from pyspark.sql import SparkSession

def f(n):
    x = random() * 2 - 1
    y = random() * 2 - 1
    return 1 if x ** 2 + y ** 2 < 1 else 0

def spark_test(spark,n):
    partitions = 40
    count = spark.sparkContext.parallelize(xrange(1, n + 1), partitions).map(f).reduce(add)
    print("Pi is roughly %f" % (4.0 * count / n))

def normal_test(n):
    count = sum(map(f,xrange(1, n + 1)))
    print("Pi is roughly %f" % (4.0 * count / n))
        
if __name__ == "__main__":
    a = open('pi_test.csv','a')
    a.write('Spark,')
    a.write('Normal\n')
    a.close()
    spark = SparkSession\
        .builder\
        .appName("PythonPi")\
        .getOrCreate()
    for n in np.arange(1000000,100000000,1000000):
        spark = SparkSession\
            .builder\
            .appName("PythonPi")\
            .getOrCreate()
        a = open('pi_test.csv','a')
        start = time.time()
        spark_test(spark,n)
        a.write(str(time.time()-start))
        a.write(',')
        #print('spark use %s s' % (time.time()-start))
        #spark.stop()
        start = time.time()
        normal_test(n)
        a.write(str(time.time()-start))
        a.write('\n')
	a.close()
        #print('normal use %s s'% (time.time()-start))
    spark.stop()
