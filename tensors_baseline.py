import numpy as np
from complition import calc_unobserved_rmse, svt_solve
from complition.evaluation import calc_unobserved_rmse2
from svp_solver import svp_solve, tensor_svp_solve
from numpy.linalg import matrix_rank

import seaborn as sb
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.random import random_tucker



size = 10
k = 5
iters = 100000
np.random.seed(101)
noise = np.random.random((size, size, size))
mask = np.round(np.random.random((size, size, size)))
tensor  =random_tucker((size,size,size), (k,k,k), full=True, orthogonal=False, random_state=np.random.RandomState(110))*10



F = tensor+noise
#F/=100


def t():
    X,Y = [],[]
    for delta in np.arange(0.8,1.9,0.1):
        F_res,error = tensor_svp_solve(F,mask,delta=delta,max_iterations=iters,k = (k,k,k),taker_iters=10000,epsilon=0.01)
        X.append(delta)
        Y.append(error[-1])

    sb.lineplot(X,Y)
    plt.show()
    exit(0)

R1 = tl.unfold(F,0)
T1 = tl.unfold(tensor,0)
def T2():

    X,Y,Y2 = [],[],[]
    for rg in range(1,size):
        F_res, error = tensor_svp_solve(F, mask, delta=1.5, max_iterations=iters, k=(rg, rg, rg), taker_iters=10000,epsilon=0.01,R = tensor)
        X.append(rg)
        R_hat1 = tl.unfold(F_res, 0)
        Y.append(calc_unobserved_rmse2(R1, R_hat1, tl.unfold(1-mask,0)))
        Y2.append(calc_unobserved_rmse2(T1, R_hat1, tl.unfold(1-mask,0)))
    sb.lineplot(X,Y)
    sb.lineplot(X,Y2)
    plt.show()
    exit(0)
#F*=100
#F_res*=100
rg = k
F_res, error,e2 = tensor_svp_solve(F, mask, delta=1.5, max_iterations=iters, k=(rg, rg, rg), taker_iters=10000,epsilon=0.0001,R = tensor)


R_hat1 = tl.unfold(F_res,0)
R_hat2 = tl.unfold(F_res,1)
R_hat3 = tl.unfold(F_res,2)

print(matrix_rank(R_hat1),matrix_rank(R_hat2),matrix_rank(R_hat3))


print(calc_unobserved_rmse2(R1, R_hat1, tl.unfold(1-mask,0)))
print(calc_unobserved_rmse2(R1, R_hat1, tl.unfold(mask,0)))

print(calc_unobserved_rmse2(tl.unfold(tensor,0), R_hat1, tl.unfold(1-mask,0)))
print(calc_unobserved_rmse2(tl.unfold(tensor,0), R_hat1, tl.unfold(mask,0)))

start = 5000
sb.lineplot(np.arange(start,iters),error[start:])
plt.show()

start = 5000
sb.lineplot(np.arange(start,iters),e2[start:])
plt.show()

start = 5000
sb.lineplot(np.arange(start,iters),error[start:])
sb.lineplot(np.arange(start,iters),e2[start:])
plt.show()

start = 50000
sb.lineplot(np.arange(start,iters),error[start:])
sb.lineplot(np.arange(start,iters),e2[start:])
plt.show()