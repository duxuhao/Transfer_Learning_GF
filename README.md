# Green Function Modelling

combine the 
- machine learning algorithm
- genetic algorithm
- experimental data
- finite element analysis</li>

Data is avabile on one drive[here](https://1drv.ms/f/s!AqlsJJroirk5hS8piJUKLfaPmduu)

to  model the Green function of complicate structure

Main codes for FEM are in folder ***Abaqus code***

Main codes for genetic algorithm are in folder ***GA***

Main codes for final model are in folder ***Experiment***

spark idea: 

test in the GA_Spark.py:
spark is not suitable for sorting as the communication time is much longer than the computation time, while when machine learning training is added, the computation time is much longer, so it is quite suitable to use spark for sorting.
And a single machine learning take quite a short time, so the spark for machine learning is useless as the communication time is much longer.