# Green Function Modelling

combine the 
- machine learning algorithm
- genetic algorithm
- experimental data
- finite element analysis</li>

to  model the Green function of complicate structure

spark idea: 

Oriental thinking: take it as a whole structure.
Western thinking: divided into basic model, limbs, yorks, etc.

test in the GA_Spark.py:
spark is not suitable for sorting as the communication time is much longer than the computation time, while when machine learning training is added, the computation time is much longer, so it is quite suitable to use spark for sorting.
And a single machine learning take quite a short time, so the spark for machine learning is useless as the communication time is much longer.

Wait for the python API of lightgbm from microsoft
