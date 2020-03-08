import numpy as np
import pandas as pd

from complition.evaluation import calc_unobserved_rmse2
from svp_solver import svp_solve
from numpy.linalg import matrix_rank

import seaborn as sb
import matplotlib.pyplot as plt


data = pd.read_csv("RecData/ratings.csv")
data = data[["userId","movieId","rating"]]
matrix = data.pivot(index='userId', columns='movieId', values='rating').values
matrix = matrix
ch = []
k = 400
for i in range(0,matrix.shape[1]):
    print(np.count_nonzero(pd.isna(matrix[:,i]).astype(int)))
    if np.count_nonzero(pd.isna(matrix[:,i]).astype(int))<k:
        ch.append(i)

print(ch)
print(matrix[:,ch].shape)
print(matrix.shape)

print(matrix[:,ch])





exit(0)
filled_matrix  = matrix[:,np.isnan(matrix[:]).shape[0]<400]
print(filled_matrix)

print(matrix.shape,filled_matrix.shape)
matrix = filled_matrix

mask = 1-np.isnan(matrix).astype(int)

matrix = np.nan_to_num(matrix)

print(np.count_nonzero(mask))

exit(0)

answ,er = svp_solve(matrix,mask,delta=0.5,max_iterations=10,k = 100)

print(answ)

print(calc_unobserved_rmse2(matrix,answ,mask))

print(np.round(answ[:10][:10]))
print(matrix[:10][:10])

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3)
