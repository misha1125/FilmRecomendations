from __future__ import division
import numpy as np
import logging

from tensorly import tucker_to_tensor
from tensorly.decomposition import tucker, non_negative_tucker


def fro_norm_2(A):
    sum = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            sum += A[i,j]**2
    return sum
def fro_norm_tensor_2(A):
    sum = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                sum += A[i,j,k]**2
    return sum

def fro_norm_tensor4_2(A):
    sum = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                for l in range(A.shape[3]):
                    sum += A[i,j,k,l]**2
    return sum

def svp_solve(A, mask, delta=0.1, epsilon=1e-2, max_iterations=1000, k=10):
    """
     A : m x n array
            матрица, которую нужно дополнить
        mask : m x n array
            матрица из 0 и 1, соответствующая диагонали P_Omega
    """
    X = np.zeros_like(A)
    error = []
    for t in range(max_iterations):
        Y = X - delta * mask * (X - A)
        U, S, V = np.linalg.svd(Y,full_matrices=False)
        S[k:] = 0
        X = np.linalg.multi_dot([U, np.diag(S),V])
        error.append(fro_norm_2(mask * (X - A)))
        if((t+1)%100==0):
            print(t)
        if fro_norm_2(mask * (X - A)) < epsilon:break
    return X,error

def tensor_svp_solve(A, mask, delta=0.1, epsilon=1e-2, max_iterations=1000, k=(10,10,10),taker_iters = 1000,R = None):
    """
     A : m x n x r тензор, который необходимо дополнить

     mask : m x n x r тензор из 0 и единиц,соответствующий отображению A(x) = mask*x
    """

    X = np.zeros_like(A)#X = 0
    t = 0
    error = []
    error2 = []
    for t in range(max_iterations):
        Y = X - delta * mask * (X - A)#вычисление Y
        #ортогональное разложение Таккера, рудукция к разложению с рангами k
        #и перемножение обратно в тензор
        X  = tucker_to_tensor(
            tucker(Y, ranks=k, n_iter_max=taker_iters, init='svd', svd='numpy_svd',tol=1e-3))
        e = fro_norm_tensor_2(mask * (X - A))

        error.append(e)
        error2.append(fro_norm_tensor_2((1-mask) * (X - A)))
        #проверка условия завершения алгоритма
        #print(t,e)
        if(e<epsilon):break
    print(t)
    return X,np.array(error),error2


