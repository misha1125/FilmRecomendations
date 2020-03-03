import numpy as np
import pandas as pd

from complition.evaluation import calc_unobserved_rmse2
from svp_solver import svp_solve
from numpy.linalg import matrix_rank

import seaborn as sb
import matplotlib.pyplot as plt

df =  pd.DataFrame()

print(df)
exit(0)
data = pd.read_csv("RecData/ratings.csv")
data = data[["userId","movieId","rating"]]
matrix = data.pivot(index='userId', columns='movieId', values='rating')

mask = 1-np.isnan(matrix).values.astype(int)

matrix = np.nan_to_num(matrix.values)




answ,er = svp_solve(matrix,mask,delta=0.5,max_iterations=10,k = 100)

print(answ)

print(calc_unobserved_rmse2(matrix,answ,mask))

print(np.round(answ[:10][:10]))
print(matrix[:10][:10])

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3)
